//
// Created by Shujian Qian on 2024-04-20.
//

//
// Created by Shujian Qian on 2023-08-08.
//

#include <iostream>
#include <memory>
#include <getopt.h>

#include "gpu_execution_planner.h"
#include "gpu_allocator.h"
#include <benchmarks/benchmark.h>
#include "benchmarks/ycsb.h"
#include "util_log.h"
#include "txn_bridge.h"

#include "benchmarks/micro.h"
#include "gacco/benchmarks/ycsb.h"

static constexpr struct option long_options[] = {{"benchmark", required_argument, nullptr, 'b'},
    {"database", required_argument, nullptr, 'd'}, {"num_warehouses", required_argument, nullptr, 'w'},
    {"skew_factor", required_argument, nullptr, 'a'}, {"fullread", required_argument, nullptr, 'r'},
    {"cpu_exec_num_threads", required_argument, nullptr, 'c'}, {"num_epochs", required_argument, nullptr, 'e'},
    {"num_txns", required_argument, nullptr, 's'}, {"split_fields", required_argument, nullptr, 'f'},
    {"commutative_ops", required_argument, nullptr, 'm'}, {"num_records", required_argument, nullptr, 'n'},
    {"exec_device", required_argument, nullptr, 'x'},
    {"abort_percentage", required_argument, nullptr, 'p'},
    {nullptr, 0, nullptr, 0}};

static char optstring[] = "b:d:w:a:r:c:e:s:f:m:n:x:p:";

int main(int argc, char **argv)
{

    epic::ycsb::YcsbConfig ycsb_config;
    epic::micro::MicroConfig micro_config;

    int retval = 0;
    char *end_char = nullptr;
    std::string bench = "micro";
    std::string db = "epic";
    bool commutative_ops = false;
    while ((retval = getopt_long(argc, argv, optstring, long_options, nullptr)) != -1)
    {
        switch (retval)
        {
        case 'b':
            bench = std::string(optarg);
            if (bench == "ycsba")
            {
                bench = "ycsb";
                ycsb_config.txn_mix = {50, 50, 0, 0};
            }
            else if (bench == "ycsbb")
            {
                bench = "ycsb";
                ycsb_config.txn_mix = {95, 5, 0, 0};
            }
            else if (bench == "ycsbc")
            {
                bench = "ycsb";
                ycsb_config.txn_mix = {100, 0, 0, 0};
            }
            else if (bench == "ycsbf")
            {
                bench = "ycsb";
                ycsb_config.txn_mix = {50, 0, 50, 0};
            }
            else if (bench == "micro")
            {
                // do nothing
            }
            else
            {

                throw std::runtime_error("Invalid benchmark name");
            }
            break;
        case 'd':
            db = std::string(optarg);
            if (db != "epic" && db != "gacco")
            {
                throw std::runtime_error("Invalid database name");
            }
            break;
        case 'w':
            errno = 0;
            break;
        case 'a':
            errno = 0;
            ycsb_config.skew_factor = strtod(optarg, &end_char);
            micro_config.skew_factor = strtod(optarg, &end_char);
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid skew factor");
            }
            break;
        case 'r':
            if (std::string(optarg) == "true")
            {
                ycsb_config.full_record_read = true;
            }
            else if (std::string(optarg) == "false")
            {
                ycsb_config.full_record_read = false;
            }
            else
            {
                throw std::runtime_error("Invalid full record read");
            }
            break;
        case 'c':
            errno = 0;
            ycsb_config.cpu_exec_num_threads = strtoul(optarg, &end_char, 0);
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of CPU execution threads");
            }
            break;
        case 'e':
            errno = 0;
            ycsb_config.epochs = strtoul(optarg, &end_char, 0);
            micro_config.epochs = ycsb_config.epochs;
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of epochs");
            }
            break;
        case 's':
            errno = 0;
            ycsb_config.num_txns = strtoul(optarg, &end_char, 0);
            micro_config.num_txns = ycsb_config.num_txns;
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of transactions");
            }
            break;
        case 'f':
            if (std::string(optarg) == "true")
            {
                ycsb_config.split_field = true;
            }
            else if (std::string(optarg) == "false")
            {
                ycsb_config.split_field = false;
            }
            else
            {
                throw std::runtime_error("Invalid split fields");
            }
            break;
        case 'm':
            if (std::string(optarg) == "true")
            {
                commutative_ops = true;
            }
            else if (std::string(optarg) == "false")
            {
                commutative_ops = false;
            }
            else
            {
                throw std::runtime_error("Invalid commutative ops");
            }
            break;
        case 'n':
            errno = 0;
            ycsb_config.num_records = strtoul(optarg, &end_char, 0);
            micro_config.num_records = ycsb_config.num_records;
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid number of records");
            }
            break;
        case 'x':
            if (std::string(optarg) == "cpu")
            {
                ycsb_config.execution_device = epic::DeviceType::CPU;
            }
            else if (std::string(optarg) == "gpu")
            {
                ycsb_config.execution_device = epic::DeviceType::GPU;
            }
            else
            {
                throw std::runtime_error("Invalid execution device");
            }
            break;
        case 'p':
            errno = 0;
            micro_config.abort_percentage = strtoul(optarg, &end_char, 0);
            if (errno != 0 || end_char == optarg || *end_char != '\0')
            {
                throw std::runtime_error("Invalid abort percentage");
            }
            break;
        default:
            throw std::runtime_error("Invalid option");
        }
    }

    /* this is a hack to run gacco NewOrder without holding locks on warehouse... */

    std::unique_ptr<epic::Benchmark> benchmark;
    if (bench == "ycsb")
    {
        if (db == "epic")
        {
            benchmark = std::make_unique<epic::ycsb::YcsbBenchmark>(ycsb_config);
        }
        else if (db == "gacco")
        {
            benchmark = std::make_unique<gacco::ycsb::YcsbBenchmark>(ycsb_config);
        }
        else
        {
            throw std::runtime_error("Invalid database name");
        }
    } else if (bench == "micro")
    {
       if (db == "epic")
       {
           benchmark = std::make_unique<epic::micro::MicroBenchmark>(micro_config);
       } else
       {
           throw std::runtime_error("Invalid database name for microbenchmark");
       }
    } else
    {
        throw std::runtime_error("Unknown workload");
    }

    benchmark->loadInitialData();
    benchmark->generateTxns();
    benchmark->runBenchmark();

    return 0;
}