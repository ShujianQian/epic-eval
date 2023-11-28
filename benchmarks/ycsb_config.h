//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_CONFIG_H
#define EPIC_BENCHMARKS_YCSB_CONFIG_H

#include <cstdint>
#include <cstddef>
#include <cassert>

#include <util_device_type.h>

namespace epic::ycsb {

struct YcsbConfig
{
    static constexpr size_t num_ops_per_txn = 10;

    struct YcsbTxnMix
    {
        size_t num_reads = num_ops_per_txn;
        size_t num_writes = 0;
        size_t num_rmw = 0;
        size_t num_inserts = 0;

        YcsbTxnMix() = default;
        YcsbTxnMix(size_t num_reads, size_t num_writes, size_t num_rmw, size_t num_inserts)
            : num_reads(num_reads)
            , num_writes(num_writes)
            , num_rmw(num_rmw)
            , num_inserts(num_inserts)
        {
            assert(num_reads + num_writes + num_rmw + num_inserts == 100);
        }
    } txn_mix;
    size_t num_records = 1'000'000;
    size_t num_txns = 100'000;
    size_t starting_num_records = 1'000'000;
    size_t epochs = 20;
    double skew_factor = 0.0;
    DeviceType index_device = DeviceType::CPU;
    DeviceType initialize_device = DeviceType::GPU;
    DeviceType execution_device = DeviceType::GPU;
    bool split_field = true;
    bool full_record_read = false;
};

} // namespace epic::ycsb

#endif // EPIC_BENCHMARKS_YCSB_CONFIG_H
