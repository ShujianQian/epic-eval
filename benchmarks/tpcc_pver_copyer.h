//
// Created by Shujian Qian on 2024-01-25.
//

#ifndef EPIC_BENCHMARKS_TPCC_PVER_COPYER_H
#define EPIC_BENCHMARKS_TPCC_PVER_COPYER_H

#include <benchmarks/tpcc_storage.h>

namespace epic::tpcc {

void copyTpccPver(
    TpccRecords records, TpccVersions versions,
    op_t *d_warehouse_ops_to_copy, uint32_t *d_warehouse_ver_to_copy, uint32_t warehouse_num_copy,
    op_t *d_district_ops_to_copy, uint32_t *d_district_ver_to_copy, uint32_t district_num_copy,
    op_t *d_customer_ops_to_copy, uint32_t *d_customer_ver_to_copy, uint32_t customer_num_copy,
    op_t *d_stock_ops_to_copy, uint32_t *d_stock_ver_to_copy, uint32_t stock_num_copy
    );

}

#endif // EPIC_BENCHMARKS_TPCC_PVER_COPYER_H
