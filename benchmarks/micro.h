//
// Created by Shujian Qian on 2024-04-20.
//

#ifndef MICRO_H
#define MICRO_H

#include <txn.h>
#include <txn_bridge.h>
#include <execution_planner.h>
#include <gpu_execution_planner.h>
#include <benchmarks/benchmark.h>
#include <benchmarks/micro_config.h>
#include <benchmarks/micro_txn.h>
#include <benchmarks/micro_storage.h>
#include <benchmarks/micro_index.h>
#include <benchmarks/micro_submitter.h>
#include <benchmarks/micro_executor.h>

namespace epic::micro {

class MicroBenchmark : public Benchmark
{
public:
    MicroConfig config;
    std::vector<TxnArray<MicroTxn>> txn_array;

    struct EpochTxnData
    {
        TxnArray<MicroTxn> index_input;
        TxnArray<MicroTxnParams> index_output, initialization_input, execution_param_input;
        bool *retry = nullptr;

        TxnBridge input_index_bridge, index_initialization_bridge, index_execution_param_bridge;
    } epoch_txn_data[2];
    TxnArray<MicroTxnExecPlan> initialization_output, execution_plan_input;
    TxnBridge initialization_execution_plan_bridge;

    MicroVersion *versions;
    MicroRecord *records;

    MicroIndex index;
    GpuTableExecutionPlanner<TxnArray<MicroTxnExecPlan>> planner;
    MicroSubmitter submitter;
    MicroExecutor executor;

    // MicroIndex idx;

    explicit MicroBenchmark(MicroConfig config);

    void loadInitialData() override;
    void generateTxns() override;
    void runBenchmark() override;
};

} // namespace epic::micro

#endif // MICRO_H
