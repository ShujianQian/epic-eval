//
// Created by Shujian Qian on 2023-11-27.
//

#include <gacco/benchmarks/ycsb.h>

#include <random>

#include <txn.h>
#include <gpu_allocator.h>
#include <util_random_zipfian.h>
#include <benchmarks/ycsb_gpu_index.h>
#include <gacco/benchmarks/ycsb_gpu_executor.h>
#include <gacco/benchmarks/ycsb_gpu_submitter.h>

namespace gacco::ycsb {

using epic::BaseTxn;
using epic::ZipfianRandom;
using epic::ycsb::YcsbGpuIndex;

YcsbBenchmark::YcsbBenchmark(YcsbConfig config)
    : config(config)
    , txn_array(config.epochs)
    , index_input(config.num_txns, config.index_device, false)
    , index_output(config.num_txns, config.index_device)
    , initialization_input(config.num_txns, config.initialize_device, false)
{

    for (int i = 0; i < config.epochs; ++i)
    {
        txn_array[i] = TxnArray<YcsbTxn>(config.num_txns, epic::DeviceType::CPU);
    }
    index = std::make_shared<YcsbGpuIndex>(config);
    input_index_bridge.Link(txn_array[0], index_input);
    index_initialization_bridge.Link(index_output, initialization_input);
    if (config.initialize_device == epic::DeviceType::GPU)
    {
        epic::GpuAllocator allocator;
        planner = std::make_shared<GpuTableExecutionPlanner>(
            "gacco ycsb planner", allocator, 0, 10, config.num_txns, config.num_records);
        planner->Initialize();
        allocator.PrintMemoryInfo();

        submitter = std::make_shared<YcsbGpuSubmitter>(
            YcsbSubmitter::TableSubmitDest{planner->d_num_ops, planner->d_op_offsets, planner->d_submitted_ops,
                planner->d_scratch_array, planner->scratch_array_bytes, planner->curr_num_ops},
            config);
    }
    else
    {
        throw std::runtime_error("Unsupported ycsb initialization device type");
    }

    if (config.execution_device == epic::DeviceType::GPU)
    {
        auto &logger = epic::Logger::GetInstance();
        logger.Info("Allocating records");
        epic::GpuAllocator allocator;
        records = static_cast<YcsbValue *>(allocator.Allocate(sizeof(YcsbValue) * config.num_records));
        executor = std::make_shared<YcsbGpuExecutor>(records, planner->table_lock, initialization_input, config);
        allocator.PrintMemoryInfo();
    }
    else
    {
        throw std::runtime_error("Unsupported ycsb execution device type");
    }
}

void YcsbBenchmark::loadInitialData()
{
    index->loadInitialData();
}
void YcsbBenchmark::generateTxns()
{
    auto &logger = epic::Logger::GetInstance();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> percentage_gen(0, 99);
    std::uniform_int_distribution<> field_gen(0, 9);
    ZipfianRandom zipf;
    zipf.init(config.num_records, config.skew_factor, rd());
    uint32_t max_existing_record = static_cast<uint32_t>(config.starting_num_records);

    auto getOpType = [&](int percentage) -> YcsbOpType {
        int acc = config.txn_mix.num_reads;
        if (percentage < acc)
        {
            return config.full_record_read ? YcsbOpType::FULL_READ : YcsbOpType::READ;
        }
        acc += config.txn_mix.num_writes;
        if (percentage < acc)
        {
            return YcsbOpType::UPDATE;
        }
        acc += config.txn_mix.num_rmw;
        if (percentage < acc)
        {
            return config.full_record_read ? YcsbOpType::FULL_READ_MODIFY_WRITE : YcsbOpType::READ_MODIFY_WRITE;
        }
        return YcsbOpType::INSERT;
    };

    for (size_t epoch = 0; epoch < config.epochs; ++epoch)
    {
        logger.Info("Generating epoch {}", epoch);
        for (size_t i = 0; i < config.num_txns; ++i)
        {
            BaseTxn *base_txn = txn_array[epoch].getTxn(i);
            uint32_t timestamp = epoch * config.num_txns + i;
            YcsbTxn *txn = reinterpret_cast<YcsbTxn *>(base_txn->data);
            for (int i = 0; i < config.num_ops_per_txn; ++i)
            {
                int percentage = percentage_gen(gen);
                txn->ops[i] = getOpType(percentage);
                if (txn->ops[i] == YcsbOpType::INSERT)
                {
                    txn->keys[i] = max_existing_record;
                    ++max_existing_record;
                    continue;
                }

                bool retry;
                do
                {
                    do
                    {
                        txn->keys[i] = zipf.next();
                    } while (txn->keys[i] >= max_existing_record);
                    retry = false;
                    for (int j = 0; j < i; ++j)
                    {
                        if (txn->keys[i] == txn->keys[j])
                        {
                            retry = true;
                            break;
                        }
                    }
                } while (retry);
                txn->fields[i] = field_gen(gen);
            }
        }
    }
}

void YcsbBenchmark::runBenchmark()
{
    auto &logger = epic::Logger::GetInstance();
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    for (uint32_t epoch_id = 1; epoch_id <= config.epochs; ++epoch_id)
    {
        logger.Info("Running epoch {}", epoch_id);
        /* transfer */
        {
            start_time = std::chrono::high_resolution_clock::now();
            uint32_t index_epoch_id = epoch_id - 1;
            input_index_bridge.Link(txn_array[index_epoch_id], index_input);
            input_index_bridge.StartTransfer();
            input_index_bridge.FinishTransfer();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} index_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* index */
        {
            start_time = std::chrono::high_resolution_clock::now();
            uint32_t index_epoch_id = epoch_id - 1;
            index->indexTxns(index_input, index_output, index_epoch_id);
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} indexing time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* transfer */
        {
            start_time = std::chrono::high_resolution_clock::now();
            index_initialization_bridge.StartTransfer();
            index_initialization_bridge.FinishTransfer();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} init_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* submit */
        {
            start_time = std::chrono::high_resolution_clock::now();
            submitter->submit(initialization_input);
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} submission time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* initialize */
        {
            start_time = std::chrono::high_resolution_clock::now();
            planner->InitializeExecutionPlan();
            planner->FinishInitialization();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} initialization time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* transfer */
        {
            start_time = std::chrono::high_resolution_clock::now();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} exec_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* execution */
        {
            start_time = std::chrono::high_resolution_clock::now();

            executor->execute(epoch_id);

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} execution time: {} us", epoch_id,
                        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }
    }
}

} // namespace gacco::ycsb