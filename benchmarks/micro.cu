//
// Created by Shujian Qian on 2024-04-20.
//

#include <random>

#include <gpu_allocator.h>
#include <util_log.h>
#include <benchmarks/micro.h>
#include <util_random_zipfian.h>

namespace epic::micro {

static GpuAllocator allocator;

MicroBenchmark::MicroBenchmark(MicroConfig config)
    : config(config)
    , index(config)
    , planner("micro_planner", allocator, 0, 20, config.num_txns * 2, config.num_records, initialization_output)
    , submitter(config)
    , executor(config)
    , txn_array(config.epochs)
{
    auto &logger = Logger::GetInstance();
    for (int i = 0; i < config.epochs; ++i)
    {
        txn_array[i] = TxnArray<MicroTxn>(config.num_txns, DeviceType::CPU);
    }
    for (auto &epoch_txn_d : epoch_txn_data)
    {
        epoch_txn_d.index_input = TxnArray<MicroTxn>(config.num_txns, DeviceType::GPU, false);
        epoch_txn_d.index_output = TxnArray<MicroTxnParams>(config.num_txns, DeviceType::GPU, true);
        epoch_txn_d.initialization_input = TxnArray<MicroTxnParams>(config.num_txns, DeviceType::GPU, false);
        epoch_txn_d.execution_param_input = TxnArray<MicroTxnParams>(config.num_txns, DeviceType::GPU, false);
        epoch_txn_d.retry = static_cast<bool *>(allocator.Allocate(config.num_txns * sizeof(bool)));
        epoch_txn_d.input_index_bridge.Link(epoch_txn_d.index_output, epoch_txn_d.initialization_input);
        epoch_txn_d.index_initialization_bridge.Link(epoch_txn_d.index_output, epoch_txn_d.initialization_input);
        epoch_txn_d.index_execution_param_bridge.Link(epoch_txn_d.index_output, epoch_txn_d.execution_param_input);
    }

    /* 2 times epoch size for retries */
    initialization_output = TxnArray<MicroTxnExecPlan>(config.num_txns * 2, DeviceType::GPU, true);
    execution_plan_input = TxnArray<MicroTxnExecPlan>(config.num_txns * 2, DeviceType::GPU, false);
    initialization_execution_plan_bridge.Link(initialization_output, execution_plan_input);

    records = static_cast<MicroRecord *>(allocator.Allocate(config.num_records * sizeof(MicroRecord)));
    versions = static_cast<MicroVersion *>(allocator.Allocate(config.num_txns * 2 * 10 * sizeof(MicroVersion)));

    planner.Initialize();
    submitter.setSubmitDest(MicroSubmitter::TableSubmitDest{planner.d_num_ops, planner.d_op_offsets,
        planner.d_submitted_ops, planner.d_scratch_array, planner.scratch_array_bytes, &planner.curr_num_ops});

}

void MicroBenchmark::loadInitialData()
{
    index.loadInitialData();
}

void MicroBenchmark::generateTxns()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> percentage_gen(0, 99);
    ZipfianRandom zipf;
    zipf.init(config.num_records, config.skew_factor, rd());

    for (size_t epoch = 0; epoch < config.epochs; ++epoch)
    {
        for (size_t i = 0; i < config.num_txns; ++i)
        {
            BaseTxn *base_txn = txn_array[epoch].getTxn(i);
            MicroTxn *txn = reinterpret_cast<MicroTxn *>(base_txn->data);
            for (int j = 0; j < 10; ++j)
            {
                bool retry = false;
                do
                {
                    txn->keys[j] = zipf.next();
                    retry = false;
                    for (int k = 0; k < j; ++k)
                    {
                        if (txn->keys[k] == txn->keys[j])
                        {
                            retry = true;
                            break;
                        }
                    }
                } while (retry);
            }
            txn->abort = percentage_gen(gen) < config.abort_percentage;
        }
    }
}


void MicroBenchmark::runBenchmark()
{
    auto &logger = Logger::GetInstance();
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    for (uint32_t epoch_id = 1; epoch_id <= config.epochs; ++epoch_id)
    {
        const uint32_t prev_data_idx = (epoch_id + 1) % 2;
        const uint32_t curr_data_idx = epoch_id % 2;
        auto &prev_epoch_data = epoch_txn_data[prev_data_idx];
        auto &curr_epoch_data = epoch_txn_data[curr_data_idx];

        logger.Info("Running epoch {}", epoch_id);

        /* transfer index data */
        {
            start_time = std::chrono::high_resolution_clock::now();
            uint32_t index_epoch_id = epoch_id - 1;
            curr_epoch_data.input_index_bridge.Link(txn_array[index_epoch_id], curr_epoch_data.index_input);
            curr_epoch_data.input_index_bridge.StartTransfer();
            curr_epoch_data.input_index_bridge.FinishTransfer();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} index_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* index */
        {
            start_time = std::chrono::high_resolution_clock::now();
            uint32_t index_epoch_id = epoch_id - 1;
            index.indexTxns(prev_epoch_data.index_input, prev_epoch_data.index_output, prev_epoch_data.retry,
                curr_epoch_data.index_input, curr_epoch_data.index_output);
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} indexing time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* transfer */
        {
            start_time = std::chrono::high_resolution_clock::now();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} init_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* submit */
        {
            start_time = std::chrono::high_resolution_clock::now();
            submitter.submit(
                prev_epoch_data.initialization_input, prev_epoch_data.retry, curr_epoch_data.initialization_input);
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} submission time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* initialize */
        {
            start_time = std::chrono::high_resolution_clock::now();
            planner.InitializeExecutionPlan();
            planner.FinishInitialization();
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

            executor.execute(records, versions, prev_epoch_data.execution_param_input, prev_epoch_data.retry,
                curr_epoch_data.execution_param_input, curr_epoch_data.retry, execution_plan_input, epoch_id);

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} execution time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }
    }
}

namespace detail {

__device__ uint32_t txn_counter = 0; /* used for scheduling txns among threads */
const uint32_t zero = 0;

}

} // namespace epic::micro