//
// Created by Shujian Qian on 2023-08-15.
//

#ifndef TPCC_H
#define TPCC_H

#include "tpcc_config.h"
#include "tpcc_common.h"
#include "tpcc_table.h"
#include "tpcc_txn.h"
#include <benchmarks/tpcc_storage.h>
#include "benchmarks/tpcc_submitter.h"

#include <memory>

#include "execution_planner.h"
#include "table.h"
#include "txn_bridge.h"
#include <benchmarks/tpcc_executor.h>
#include <benchmarks/tpcc_gpu_executor.h>

namespace epic::tpcc {

class TpccDb
{
private:
    TableExecutionPlanner *planner;

    Table warehouse_table;

    TpccConfig config;
    TxnInputArray<TpccTxn> txn_array;
    TxnArray<TpccTxnParam> index_output;
    TxnArray<TpccTxnParam> initialization_input;
    TxnArray<TpccExecPlan> initialization_output;

    TxnBridge index_initialization_bridge;

    TpccIndex index;
    TpccLoader loader;

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

    TpccRecords records;
    TpccVersions versions;

    std::shared_ptr<Executor> executor;
public:
    explicit TpccDb(TpccConfig config);

    void loadInitialData();
    void generateTxns();
    void runBenchmark();

    void indexEpoch(uint32_t epoch_id);

private:
};

} // namespace epic::tpcc

#endif // TPCC_H
