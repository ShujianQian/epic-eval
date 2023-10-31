//
// Created by Shujian Qian on 2023-08-08.
//

#include <iostream>

#include "gpu_execution_planner.h"
#include "gpu_allocator.h"
#include "benchmarks/ycsb.h"
#include "util_log.h"
#include "txn_bridge.h"

#include "benchmarks/tpcc.h"

int main(int argc, char **argv)
{

    epic::tpcc::TpccConfig config;
    config.epochs = 5;
    config.num_warehouses = 1;
    //    config.num_txns = 10;

    epic::tpcc::TpccDb db(config);
    db.loadInitialData();
    db.generateTxns();
    db.runBenchmark();

    return 0;
}