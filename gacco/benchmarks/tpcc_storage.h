//
// Created by Shujian Qian on 2023-11-14.
//

#ifndef EPIC_GACCO_BENCHMARKS_TPCC_STORAGE_H
#define EPIC_GACCO_BENCHMARKS_TPCC_STORAGE_H

#include <gacco/benchmarks/tpcc_table.h>

namespace gacco::tpcc {

struct TpccRecords
{
    WarehouseValue *warehouse_record = nullptr;
    DistrictValue *district_record = nullptr;
    CustomerValue *customer_record = nullptr;
    HistoryValue *history_record = nullptr;
    NewOrderValue *new_order_record = nullptr;
    OrderValue *order_record = nullptr;
    OrderLineValue *order_line_record = nullptr;
    ItemValue *item_record = nullptr;
    StockValue *stock_record = nullptr;
};

}

#endif // EPIC_GACCO_BENCHMARKS_TPCC_STORAGE_H
