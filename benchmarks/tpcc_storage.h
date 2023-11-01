//
// Created by Shujian Qian on 2023-10-31.
//

#ifndef TPCC_STORAGE_H
#define TPCC_STORAGE_H

#include <benchmarks/tpcc_table.h>
#include <storage.h>

namespace epic::tpcc {

struct TpccVersions
{
    Version<WarehouseValue> *warehouse_version = nullptr;
    Version<DistrictValue> *district_version = nullptr;
    Version<CustomerValue> *customer_version = nullptr;
    Version<HistoryValue> *history_version = nullptr;
    Version<NewOrderValue> *new_order_version = nullptr;
    Version<OrderValue> *order_version = nullptr;
    Version<OrderLineValue> *order_line_version = nullptr;
    Version<ItemValue> *item_version = nullptr;
    Version<StockValue> *stock_version = nullptr;
};

struct TpccRecords
{
    Record<WarehouseValue> *warehouse_record = nullptr;
    Record<DistrictValue> *district_record = nullptr;
    Record<CustomerValue> *customer_record = nullptr;
    Record<HistoryValue> *history_record = nullptr;
    Record<NewOrderValue> *new_order_record = nullptr;
    Record<OrderValue> *order_record = nullptr;
    Record<OrderLineValue> *order_line_record = nullptr;
    Record<ItemValue> *item_record = nullptr;
    Record<StockValue> *stock_record = nullptr;
};

} // namespace epic

#endif // TPCC_STORAGE_H
