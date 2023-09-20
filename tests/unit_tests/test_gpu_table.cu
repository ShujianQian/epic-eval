//
// Created by Shujian Qian on 2023-08-08.
//

#include "gpu_execution_planner.cuh"
#include "benchmarks/gpu_ycsb.cuh"
#include "gpu_allocator.cuh"
#include "benchmarks/ycsb.h"
#include "util_log.h"

#include "benchmarks/tpcc.h"

int main(int argc, char **argv)
{
    epic::GpuAllocator allocator;
    epic::GpuExecutionPlanner table("table1", allocator, 1000, 10, 100'000, 1'000'000);
    table.Initialize();

    epic::tpcc::TpccConfig config;
    epic::tpcc::TpccDb db(config);
    db.loadInitialData();
    db.generateTxns();

    return 0;
}