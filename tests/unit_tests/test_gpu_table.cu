//
// Created by Shujian Qian on 2023-08-08.
//

#include <iostream>

#include "gpu_execution_planner.h"
#include "benchmarks/gpu_ycsb.cuh"
#include "gpu_allocator.h"
#include "benchmarks/ycsb.h"
#include "util_log.h"
#include "txn_bridge.h"
#include "gpu_txn.cuh"

#include "benchmarks/tpcc.h"

int main(int argc, char **argv)
{
//    epic::GpuAllocator allocator;
//    epic::GpuTableExecutionPlanner table("table1", allocator, 1000, 10, 100'000, 1'000'000);
//    table.Initialize();
//    gpu_err_check(cudaSetDeviceFlags(cudaDeviceScheduleYield));

    epic::tpcc::TpccConfig config;
    config.epochs = 5;
    config.num_warehouses = 1;
//    config.num_txns = 10;
    epic::tpcc::TpccDb db(config);
//    auto db2 = new epic::tpcc::TpccDb(config);
//    db2->loadInitialData();
//    db2->generateTxns();
//    db2->runBenchmark();
    db.loadInitialData();
    db.generateTxns();
    db.runBenchmark();


    return 0;
}