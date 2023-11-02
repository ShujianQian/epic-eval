//
// Created by Shujian Qian on 2023-09-20.
//

#ifndef TPCC_CONFIG_H
#define TPCC_CONFIG_H

#include <cstdint>
#include <cstdlib>

#include "util_device_type.h"

namespace epic::tpcc {

struct TpccTxnMix
{
    uint32_t new_order = 50;
    uint32_t payment = 50;
    uint32_t order_status = 0;
    uint32_t delivery = 0;
    uint32_t stock_level = 0;

    TpccTxnMix() = default;
    TpccTxnMix(uint32_t new_order, uint32_t payment, uint32_t order_status, uint32_t delivery, uint32_t stock_level);
};

struct TpccConfig
{
    TpccTxnMix txn_mix;
    size_t num_txns = 100'000;
    size_t epochs = 20;
    size_t num_warehouses = 8;
    size_t order_table_size = 1'000'000;
    size_t orderline_table_size = 10'000'000;
    DeviceType index_device = DeviceType::CPU;
    DeviceType initialize_device = DeviceType::GPU;
    DeviceType execution_device = DeviceType::GPU;

    size_t warehouseTableSize() const
    {
        return num_warehouses * 2;
    }
    size_t districtTableSize() const
    {
        return num_warehouses * 2 * 20;
    }
    size_t customerTableSize() const
    {
        return num_warehouses * 2 * 20 * 96'000;
    }
    size_t historyTableSize() const
    {
        return num_warehouses * 2 * 20 * 96'000 * num_warehouses * 2 * 20;
    }
    size_t newOrderTableSize() const
    {
        return order_table_size;
    }
    size_t orderTableSize() const
    {
        return order_table_size;
    }
    size_t orderLineTableSize() const
    {
        return orderline_table_size;
    }
    size_t itemTableSize() const
    {
        return 200'000;
    }
    size_t stockTableSize() const
    {
        return 200'000 * num_warehouses * 2;
    }
};

} // namespace epic::tpcc

#endif // TPCC_CONFIG_H
