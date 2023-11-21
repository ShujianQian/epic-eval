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
#include "gacco/benchmarks/tpcc.h"

int main(int argc, char **argv)
{

    epic::tpcc::TpccConfig config;
    config.epochs = 5;
    config.num_warehouses = 64;
    config.txn_mix = {50, 50, 0, 0, 0};
    //        config.txn_mix = {0, 100, 0, 0, 0};
    //        config.txn_mix = {100, 0, 0, 0, 0};

    //    config.gacco_use_atomic = true;
    //    config.gacco_tpcc_stock_use_atomic = true;
    config.index_device = epic::DeviceType::GPU;
    //    config.num_txns = 100;

    epic::tpcc::TpccDb db(config);
    //        gacco::tpcc::TpccDb db(config);
    db.loadInitialData();
    db.generateTxns();
    db.runBenchmark();

    return 0;
}