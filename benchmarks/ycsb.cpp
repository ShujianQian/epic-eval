//
// Created by Shujian Qian on 2023-08-23.
//

#include <random>
#include <util_log.h>
#include <benchmarks/ycsb.h>
#include <benchmarks/ycsb_txn.h>
#include <util_random_zipfian.h>
#include <benchmarks/ycsb_index.h>
#include <benchmarks/ycsb_gpu_index.h>
#include <gpu_allocator.h>
#include <gpu_execution_planner.h>
#include <benchmarks/ycsb_gpu_submitter.h>
#include <benchmarks/ycsb_gpu_executor.h>
#include <benchmarks/ycsb_cpu_executor.h>
#include <benchmarks/ycsb_pver_copyer.h>

namespace epic::ycsb {

YcsbBenchmark::YcsbBenchmark(YcsbConfig config)
    : config(config)
    , txn_array(config.epochs)
    , index_input(config.num_txns, config.index_device, false)
    , index_output(config.num_txns, config.index_device)
    , initialization_input(config.num_txns, config.initialize_device, false)
    , initialization_output(config.num_txns, config.initialize_device)
    , execution_param_input(config.num_txns, config.execution_device, false)
    , execution_plan_input(config.num_txns, config.execution_device, false)
{
    auto &logger = Logger::GetInstance();
    for (int i = 0; i < config.epochs; ++i)
    {
        txn_array[i] = TxnArray<YcsbTxn>(config.num_txns, DeviceType::CPU);
    }
    index = std::make_shared<YcsbGpuIndex>(config);
    input_index_bridge.Link(txn_array[0], index_input);
    index_initialization_bridge.Link(index_output, initialization_input);
    index_execution_param_bridge.Link(index_output, execution_param_input);
    initialization_execution_plan_bridge.Link(initialization_output, execution_plan_input);

    GpuAllocator allocator;
    planner = std::make_shared<GpuTableExecutionPlanner>(
        "planner", allocator, 0, 11 * 10, config.num_txns, config.num_records, initialization_output);
    planner->Initialize();

    submitter = std::make_shared<YcsbGpuSubmitter>(
        YcsbSubmitter::TableSubmitDest{planner->d_num_ops, planner->d_op_offsets, planner->d_submitted_ops,
            planner->d_scratch_array, planner->scratch_array_bytes, planner->curr_num_ops},
        config);

    if (config.execution_device == DeviceType::GPU)
    {
        if (config.split_field)
        {
            size_t record_size = sizeof(YcsbFieldRecords) * config.num_records * 10;
            records = static_cast<YcsbFieldRecords *>(allocator.Allocate(record_size));
            logger.Info("Field-split record size: {}", formatSizeBytes(record_size));
            size_t version_size = sizeof(YcsbFieldVersions) * config.num_records * 10;
            versions = static_cast<YcsbFieldVersions *>(allocator.Allocate(version_size));
            logger.Info("Field-split version size: {}", formatSizeBytes(version_size));
        }
        else
        {
            size_t record_size = sizeof(YcsbRecords) * config.num_records;
            records = static_cast<YcsbRecords *>(allocator.Allocate(record_size));
            logger.Info("Record size: {}", formatSizeBytes(record_size));
            size_t version_size = sizeof(YcsbVersions) * config.num_records;
            versions = static_cast<YcsbVersions *>(allocator.Allocate(version_size));
            logger.Info("Version size: {}", formatSizeBytes(version_size));
        }
        allocator.PrintMemoryInfo();

        executor =
            std::make_shared<GpuExecutor>(records, versions, execution_param_input, execution_plan_input, config);
        //        executor =
        //            std::make_shared<GpuExecutor>(records, versions, initialization_input, initialization_output,
        //            config);
    }
    else if (config.execution_device == DeviceType::CPU)
    {
        if (config.split_field)
        {
            throw std::runtime_error(
                "epic::ycsb::YcsbBenchmark::YcsbBenchmark() found split field not supported on CPU.");
        }
        else
        {
            size_t record_size = sizeof(YcsbRecords) * config.num_records;
            records = static_cast<YcsbRecords *>(Malloc(record_size));
            logger.Info("Record size: {}", formatSizeBytes(record_size));
            size_t version_size = sizeof(YcsbVersions) * config.num_records;
            versions = static_cast<YcsbVersions *>(Malloc(version_size));
            logger.Info("Version size: {}", formatSizeBytes(version_size));
        }

        executor =
            std::make_shared<CpuExecutor>(records, versions, execution_param_input, execution_plan_input, config);
    }
    else
    {
        throw std::runtime_error("epic::ycsb::YcsbBenchmark::YcsbBenchmark() found unknown execution device.");
    }
}

void YcsbBenchmark::loadInitialData()
{
    index->loadInitialData();
}

void YcsbBenchmark::generateTxns()
{
    auto &logger = Logger::GetInstance();
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
        uint32_t num_contended = 0;
        for (size_t txn_idx = 0; txn_idx < config.num_txns; ++txn_idx)
        {
            BaseTxn *base_txn = txn_array[epoch].getTxn(txn_idx);
            uint32_t timestamp = epoch * config.num_txns + txn_idx;
            YcsbTxn *txn = reinterpret_cast<YcsbTxn *>(base_txn->data);
            constexpr uint32_t contended_threshold = 36;
//            constexpr uint32_t contended_threshold = 1700;
            for (int piece_idx = 0; piece_idx < config.num_ops_per_txn; ++piece_idx)
            {
                int percentage = percentage_gen(gen);
                txn->ops[piece_idx] = getOpType(percentage);
                if (txn->ops[piece_idx] == YcsbOpType::INSERT)
                {
                    txn->keys[piece_idx] = max_existing_record;
                    ++max_existing_record;
                    continue;
                }

                bool retry;
                do
                {
                    do
                    {
                        txn->keys[piece_idx] = zipf.next();
                    } while (txn->keys[piece_idx] >= max_existing_record);
                    retry = false;
#define EPIC_PACKED_TXN_GEN
//#undef EPIC_PACKED_TXN_GEN
#ifdef EPIC_PACKED_TXN_GEN
                    int epoch_piece_idx = txn_idx * config.num_ops_per_txn + piece_idx;
                    if ((epoch_piece_idx / 4) % 4 == 0)
                    {
                        retry |= txn->keys[piece_idx] >= contended_threshold;
                    } else {
                        retry |= txn->keys[piece_idx] < contended_threshold;
                    }
#endif
                    for (int j = 0; j < piece_idx; ++j)
                    {
                        if (txn->keys[piece_idx] == txn->keys[j])
                        {
                            retry = true;
                            break;
                        }
                    }
                } while (retry);
                if (txn->keys[piece_idx] < contended_threshold) {
                    ++num_contended;
                }
                txn->fields[piece_idx] = field_gen(gen);
            }
        }
        logger.Info("Epoch {} num_100: {}", epoch, num_contended);
    }

#if 1 // Print Txn Key Hist
    {
        uint32_t key_hist[32]{};

        const uint32_t epoch = 0;
        for (size_t txn_idx = 0; txn_idx < config.num_txns; ++txn_idx)
        {
            BaseTxn *base_txn = txn_array[epoch].getTxn(txn_idx);
            YcsbTxn *txn = reinterpret_cast<YcsbTxn *>(base_txn->data);
            for (int piece_idx = 0; piece_idx < config.num_ops_per_txn; ++piece_idx)
            {
                uint32_t key = txn->keys[piece_idx];
                uint32_t idx = 0;
                uint32_t mult = 1;
                while (key >= mult)
                {
                    ++idx;
                    mult *= 2;
                }
                ++key_hist[idx];
            }
        }
        for (uint32_t i = 0; i < 30; ++i)
        {
            logger.Info("key_hist[{}-{}]: {}", (1<<i) / 2, 1<<i, key_hist[i]);
        }
    }

#endif
}
void YcsbBenchmark::runBenchmark()
{

    auto &logger = Logger::GetInstance();
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
            index_execution_param_bridge.StartTransfer();
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
            initialization_execution_plan_bridge.StartTransfer();
            index_execution_param_bridge.FinishTransfer();
            initialization_execution_plan_bridge.FinishTransfer();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} exec_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* execution */
        {
            start_time = std::chrono::high_resolution_clock::now();

            executor->execute(epoch_id, planner->d_pver_sync_expect, planner->d_pver_sync_counter);

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} execution time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* pver copy */
        {
            if (config.use_copy_single_version)
            {
                start_time = std::chrono::high_resolution_clock::now();
                copyYcsbPver(std::get<YcsbRecords *>(records), std::get<YcsbVersions *>(versions),
                    static_cast<op_t *>(planner->d_permenant_version_rids), nullptr, planner->curr_num_ops);
                end_time = std::chrono::high_resolution_clock::now();
                logger.Info("Epoch {} copy pver time: {} us", epoch_id,
                    std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
            }
        }

        /* sync single version */
        {
            /* print sync timing stats */
            logger.Info("Epoch {} sync single version stats:", epoch_id);
            executor->printStat();

        }
    }
}
} // namespace epic::ycsb