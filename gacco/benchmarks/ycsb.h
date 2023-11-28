//
// Created by Shujian Qian on 2023-11-27.
//

#ifndef EPIC_GACCO_BENCHMARKS_YCSB_H
#define EPIC_GACCO_BENCHMARKS_YCSB_H

#include <vector>

#include <txn.h>
#include <txn_bridge.h>
#include <benchmarks/benchmark.h>
#include <benchmarks/ycsb_txn.h>
#include <benchmarks/ycsb_table.h>
#include <benchmarks/ycsb_index.h>
#include <benchmarks/ycsb_config.h>
#include <gacco/gpu_execution_planner.h>
#include <gacco/benchmarks/ycsb_executor.h>
#include <gacco/benchmarks/ycsb_submitter.h>

namespace gacco::ycsb {

using epic::TxnArray;
using epic::TxnBridge;
using epic::TxnInputArray;
using epic::ycsb::YcsbConfig;
using epic::ycsb::YcsbIndex;
using epic::ycsb::YcsbTxn;
using epic::ycsb::YcsbTxnParam;
using epic::ycsb::YcsbValue;

class YcsbBenchmark : public epic::Benchmark
{
public:
    YcsbConfig config;

    std::vector<TxnArray<YcsbTxn>> txn_array;
    TxnArray<YcsbTxn> index_input;
    TxnArray<YcsbTxnParam> index_output;
    TxnArray<YcsbTxnParam> initialization_input;

    TxnBridge input_index_bridge;
    TxnBridge index_initialization_bridge;

    std::shared_ptr<YcsbIndex> index;

    std::shared_ptr<TableExecutionPlanner> planner;
    std::shared_ptr<YcsbSubmitter> submitter;

    YcsbValue *records;

    std::shared_ptr<Executor> executor;

    explicit YcsbBenchmark(YcsbConfig config);
    void loadInitialData() override;
    void generateTxns() override;
    void runBenchmark() override;

    void indexEpoch(uint32_t epoch_id){/* TODO: remove */};
};

} // namespace gacco::ycsb

#endif // EPIC_GACCO_BENCHMARKS_YCSB_H
