//
// Created by Shujian Qian on 2023-08-08.
//

#include <iostream>
#include <memory>

#include "gpu_execution_planner.h"
#include "gpu_allocator.h"
#include <benchmarks/benchmark.h>
#include "benchmarks/ycsb.h"
#include "util_log.h"
#include "txn_bridge.h"

#include "benchmarks/tpcc.h"
#include "gacco/benchmarks/tpcc.h"

int main(int argc, char **argv)
{

    epic::tpcc::TpccConfig tpcc_config;
    epic::ycsb::YcsbConfig ycsb_config;
    //    ycsb_config.txn_mix = {50, 50, 0, 0};
    ycsb_config.txn_mix = {95, 5, 0, 0};
    ycsb_config.index_device = epic::DeviceType::GPU;
    ycsb_config.num_records = 2'500'000;
    ycsb_config.starting_num_records = 2'000'000;
    //        ycsb_config.skew_factor = 0.0;
    ycsb_config.skew_factor = 0.99;
    ycsb_config.epochs = 5;
    ycsb_config.split_field = false;
    ycsb_config.full_record_read = true;

    tpcc_config.epochs = 5;
    tpcc_config.num_warehouses = 64;
    //    tpcc_config.txn_mix = {50, 50, 0, 0, 0};
    //    tpcc_config.txn_mix = {0, 100, 0, 0, 0};
    tpcc_config.txn_mix = {100, 0, 0, 0, 0};

    tpcc_config.gacco_use_atomic = true;
    tpcc_config.gacco_tpcc_stock_use_atomic = true;
    tpcc_config.index_device = epic::DeviceType::GPU;
    tpcc_config.num_txns = 32768;

    //    std::unique_ptr<epic::Benchmark> benchmark = std::make_unique<gacco::tpcc::TpccDb>(tpcc_config);
    //    std::unique_ptr<epic::Benchmark> benchmark = std::make_unique<epic::tpcc::TpccDb>(tpcc_config);
    std::unique_ptr<epic::Benchmark> benchmark = std::make_unique<epic::ycsb::YcsbBenchmark>(ycsb_config);
    benchmark->loadInitialData();
    benchmark->generateTxns();
    benchmark->runBenchmark();

    return 0;
}