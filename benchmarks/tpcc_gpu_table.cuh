//
// Created by Shujian Qian on 2023-11-21.
//

#ifndef EPIC_BENCHMARKS_TPCC_GPU_TABLE_CUH
#define EPIC_BENCHMARKS_TPCC_GPU_TABLE_CUH

#include <benchmarks/tpcc_table.h>

namespace epic::tpcc::gpu {

union WarehouseKey
{
    using baseType = typename ChooseBitfieldBaseType<2 * kMaxWarehouses>::type;
    struct
    {
        baseType w_id : ceilLog2(2 * kMaxWarehouses);
    } key;
    baseType base_key;
    explicit WarehouseKey(baseType w_id)
    {
        base_key = 0;
        key.w_id = w_id;
    }
};

union DistrictKey
{
    using baseType = ChooseBitfieldBaseType<20, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType d_id : ceilLog2(20);
        baseType d_w_id : ceilLog2(2 * kMaxWarehouses);
    } key;
    baseType base_key;
    DistrictKey(baseType d_id, baseType d_w_id)
    {
        base_key = 0;
        key.d_id = d_id;
        key.d_w_id = d_w_id;
    }
};

union CustomerKey
{
    using baseType = ChooseBitfieldBaseType<96'000, 20, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType c_id : ceilLog2(96'000);
        baseType c_d_id : ceilLog2(20);
        baseType c_w_id : ceilLog2(2 * kMaxWarehouses);
    } key;
    baseType base_key;
    CustomerKey(baseType c_id, baseType c_d_id, baseType c_w_id)
    {
        base_key = 0;
        key.c_id = c_id;
        key.c_d_id = c_d_id;
        key.c_w_id = c_w_id;
    }
};


union StockKey
{
    using baseType = ChooseBitfieldBaseType<200'000, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType s_i_id : ceilLog2(200'000);
        baseType s_w_id : ceilLog2(2 * kMaxWarehouses);
    } key;
    baseType base_key;
    StockKey(baseType s_i_id, baseType s_w_id)
    {
        base_key = 0;
        key.s_i_id = s_i_id;
        key.s_w_id = s_w_id;
    }
};


}

#endif // EPIC_BENCHMARKS_TPCC_GPU_TABLE_CUH
