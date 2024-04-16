//
// Created by Shujian Qian on 2023-11-02.
//

#ifndef EPIC_GACCO_BENCHMARKS_TPCC_H
#define EPIC_GACCO_BENCHMARKS_TPCC_H

#include <vector>

#include <txn.h>
#include <benchmarks/benchmark.h>
#include <benchmarks/tpcc_config.h>
#include <benchmarks/tpcc_txn.h>
#include <benchmarks/tpcc_index.h>
#include <benchmarks/tpcc_table.h>
#include <txn_bridge.h>
#include <gacco/gpu_execution_planner.h>
#include <gacco/benchmarks/tpcc_submitter.h>
#include <gacco/benchmarks/tpcc_executor.h>


namespace gacco::tpcc {

using epic::tpcc::TpccConfig;
using epic::TxnArray;
using epic::PackedTxnArray;
using epic::TxnInputArray;
using epic::tpcc::TpccTxn;
using epic::tpcc::TpccTxnParam;
using epic::tpcc::TpccTxnParam;
using epic::tpcc::TpccIndex;
using epic::tpcc::TpccCpuIndex;
using epic::TxnBridge;
using epic::PackedTxnBridge;
using epic::tpcc::TpccPackedTxnArrayBuilder;

class TpccDb : public epic::Benchmark
{
public:
    TpccConfig config;

    std::vector<PackedTxnArray<TpccTxn>> txn_array;
    PackedTxnArray<TpccTxn> index_input;
    PackedTxnArray<TpccTxnParam> index_output;
    PackedTxnArray<TpccTxnParam> initialization_input;

    PackedTxnBridge input_index_bridge;
    PackedTxnBridge index_initialization_bridge;

    std::shared_ptr<TpccIndex<epic::tpcc::TpccTxnArrayT, epic::tpcc::TpccTxnParamArrayT>> index;

    std::shared_ptr<TableExecutionPlanner> warehouse_planner;
    std::shared_ptr<TableExecutionPlanner> district_planner;
    std::shared_ptr<TableExecutionPlanner> customer_planner;
    std::shared_ptr<TableExecutionPlanner> history_planner;
    std::shared_ptr<TableExecutionPlanner> new_order_planner;
    std::shared_ptr<TableExecutionPlanner> order_planner;
    std::shared_ptr<TableExecutionPlanner> order_line_planner;
    std::shared_ptr<TableExecutionPlanner> item_planner;
    std::shared_ptr<TableExecutionPlanner> stock_planner;
    std::shared_ptr<TpccSubmitter> submitter;

    std::shared_ptr<Executor> executor;

    TpccPackedTxnArrayBuilder builder;

    explicit TpccDb(TpccConfig config);
    void loadInitialData() override;
    void generateTxns() override;
    void runBenchmark() override;

    void indexEpoch(uint32_t epoch_id);
};

}

#endif // EPIC_GACCO_BENCHMARKS_TPCC_H
