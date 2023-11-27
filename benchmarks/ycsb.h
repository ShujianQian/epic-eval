//
// Created by Shujian Qian on 2023-08-08.
//

#ifndef YCSB_H
#define YCSB_H

#include <cstdint>
#include <vector>
#include <memory>
#include <variant>

#include <txn.h>
#include <txn_bridge.h>
#include <execution_planner.h>
#include <benchmarks/benchmark.h>
#include <benchmarks/ycsb_config.h>
#include <benchmarks/ycsb_txn.h>
#include <benchmarks/ycsb_storage.h>
#include <benchmarks/ycsb_index.h>
#include <benchmarks/ycsb_submitter.h>
#include <benchmarks/ycsb_executor.h>

namespace epic::ycsb {

class YcsbBenchmark : public Benchmark
{
public:
    YcsbConfig config;

    std::vector<TxnArray<YcsbTxn>> txn_array;
    TxnArray<YcsbTxn> index_input;
    TxnArray<YcsbTxnParam> index_output;
    TxnArray<YcsbTxnParam> initialization_input;
    TxnArray<YcsbExecPlan> initialization_output;

    TxnBridge input_index_bridge;
    TxnBridge index_initialization_bridge;

    std::shared_ptr<YcsbIndex> index;
    std::shared_ptr<TableExecutionPlanner> planner;
    std::shared_ptr<YcsbSubmitter> submitter;

    YcsbVersionArrType versions;
    YcsbRecordArrType records;

    std::shared_ptr<Executor> executor;

    explicit YcsbBenchmark(YcsbConfig config);
    ~YcsbBenchmark() override = default;

    void loadInitialData() override;
    void generateTxns() override;
    void runBenchmark() override;
};

} // namespace epic::ycsb

#endif // YCSB_H
