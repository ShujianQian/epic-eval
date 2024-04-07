//
// Created by Shujian Qian on 2023-08-28.
//

#ifndef TPCC_TABLE_H
#define TPCC_TABLE_H

#include <cstdint>
#include <memory>

#include "tpcc_config.h"
#include "tpcc_common.h"
#include "tpcc_txn.h"
#include "util_math.h"
#include "util_bitfield.h"

#include "unordered_index.h"

namespace epic::tpcc {

/* NOTE:
 * for ordered keys, it's important that the key fields are in reverse order of th key on x64 (little endian)
 * so that the key can be compared by comparing the base_key
 */

union WarehouseKey
{
    using baseType = typename ChooseBitfieldBaseType<2 * kMaxWarehouses>::type;
    struct
    {
        baseType w_id : ceilLog2(2 * kMaxWarehouses);
    } key;
    baseType base_key = 0;
    WarehouseKey() = default;
    explicit WarehouseKey(baseType w_id)
    {
        base_key = 0;
        key.w_id = w_id;
    }
};

struct WarehouseValue
{
    uint32_t placeholder;
    uint32_t w_ytd;
    uint32_t w_tax;
    uint8_t w_name[12];
    uint8_t w_street_1[20];
    uint8_t w_street_2[20];
    uint8_t w_city[20];
    uint8_t w_state[4];
    uint8_t w_zip[12];
};
static_assert(sizeof(WarehouseValue) <= 128);

union DistrictKey
{
    using baseType = ChooseBitfieldBaseType<20, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType d_id : ceilLog2(20);
        baseType d_w_id : ceilLog2(2 * kMaxWarehouses);
    } key;
    baseType base_key = 0;
    DistrictKey() = default;
    DistrictKey(baseType d_id, baseType d_w_id)
    {
        base_key = 0;
        key.d_id = d_id;
        key.d_w_id = d_w_id;
    }
};

struct DistrictValue
{
    uint32_t d_ytd;
    uint32_t d_tax;
    uint32_t d_next_o_id;
    uint8_t d_name[12];
    uint8_t d_street_1[20];
    uint8_t d_street_2[20];
    uint8_t d_city[20];
    uint8_t d_state[4];
    uint8_t d_zip[12];
};
static_assert(sizeof(DistrictValue) <= 128);

union CustomerKey
{
    using baseType = ChooseBitfieldBaseType<96'000, 20, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType c_id : ceilLog2(96'000);
        baseType c_d_id : ceilLog2(20);
        baseType c_w_id : ceilLog2(2 * kMaxWarehouses);
    } key;
    baseType base_key = 0;
    CustomerKey() = default;
    CustomerKey(baseType c_id, baseType c_d_id, baseType c_w_id)
    {
        base_key = 0;
        key.c_id = c_id;
        key.c_d_id = c_d_id;
        key.c_w_id = c_w_id;
    }
};

struct CustomerValue
{
    uint32_t c_balance;
    uint32_t c_ytd_payment;
    uint32_t c_payment_cnt;
    uint32_t c_delivery_cnt;
    uint32_t c_discount;
    uint32_t c_credit_lim;
    uint8_t c_credit[4];
};
static_assert(sizeof(CustomerValue) <= 128);

struct CustomerInfoValue
{
    uint8_t c_last[16];
    uint8_t c_first[16];
    uint8_t c_street_1[20];
    uint8_t c_street_2[20];
    uint8_t c_city[20];
    uint8_t c_state[4];
    uint8_t c_zip[12];
    uint8_t c_phone[16];
    uint32_t c_since;
    uint8_t c_middle[2];
    uint8_t c_data[500];
};

union HistoryKey
{
    using baseType = ChooseBitfieldBaseType<96'000, 20, 2 * kMaxWarehouses, 20, 2 * kMaxWarehouses, 1'000'000>::type;
    struct
    {
        baseType h_c_id : ceilLog2(96'000);
        baseType h_c_d_id : ceilLog2(20);
        baseType h_c_w_id : ceilLog2(2 * kMaxWarehouses);
        baseType h_d_id : ceilLog2(20);
        baseType h_w_id : ceilLog2(2 * kMaxWarehouses);
        baseType h_date : ceilLog2(1'000'000);
    } key;
    baseType base_key = 0;
    HistoryKey() = default;
    HistoryKey(baseType h_c_id, baseType h_c_d_id, baseType h_c_w_id, baseType h_d_id, baseType h_w_id, baseType h_date)
    {
        base_key = 0;
        key.h_c_id = h_c_id;
        key.h_c_d_id = h_c_d_id;
        key.h_c_w_id = h_c_w_id;
        key.h_d_id = h_d_id;
        key.h_w_id = h_w_id;
        key.h_date = h_date;
    }
};

struct HistoryValue
{
    uint32_t h_amount;
    uint8_t h_data[24];
};
static_assert(sizeof(HistoryValue) <= 128);

union NewOrderKey
{
    using baseType = ChooseBitfieldBaseType<10'000'000, 20, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType no_o_id : ceilLog2(10'000'000);
        baseType no_d_id : ceilLog2(20);
        baseType no_w_id : ceilLog2(2 * kMaxWarehouses);
    };
    baseType base_key = 0;
    NewOrderKey() = default;
    NewOrderKey(baseType no_o_id, baseType no_d_id, baseType no_w_id)
    {
        base_key = 0;
        this->no_o_id = no_o_id;
        this->no_d_id = no_d_id;
        this->no_w_id = no_w_id;
    }
};

struct NewOrderValue
{
    uint32_t dummy;
};

union OrderKey
{
    using baseType = typename ChooseBitfieldBaseType<10'000'000, 20, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType o_id : ceilLog2(10'000'000);
        baseType o_d_id : ceilLog2(20);
        baseType o_w_id : ceilLog2(2 * kMaxWarehouses);
    };
    baseType base_key = 0;
    OrderKey() = default;
    OrderKey(baseType o_id, baseType o_d_id, baseType o_w_id)
    {
        base_key = 0;
        this->o_id = o_id;
        this->o_d_id = o_d_id;
        this->o_w_id = o_w_id;
    }
};

struct OrderValue
{
    uint32_t o_c_id;
    uint32_t o_carrier_id;
    uint32_t o_ol_cnt;
    uint32_t o_all_local;
    uint32_t o_entry_d;
};
static_assert(sizeof(OrderValue) <= 128);

union OrderLineKey
{
    using baseType = ChooseBitfieldBaseType<10'000'000, 20, 2 * kMaxWarehouses, 15>::type;
    struct
    {
        baseType ol_number : ceilLog2(15);
        baseType ol_o_id : ceilLog2(10'000'000);
        baseType ol_d_id : ceilLog2(20);
        baseType ol_w_id : ceilLog2(2 * kMaxWarehouses);
    };
    baseType base_key = 0;
    OrderLineKey() = default;
    OrderLineKey(baseType ol_o_id, baseType ol_d_id, baseType ol_w_id, baseType ol_number)
    {
        base_key = 0;
        this->ol_o_id = ol_o_id;
        this->ol_d_id = ol_d_id;
        this->ol_w_id = ol_w_id;
        this->ol_number = ol_number;
    }
};

struct OrderLineValue
{
    uint32_t ol_i_id;
    uint32_t ol_delivery_d;
    uint32_t ol_amount;
    uint32_t ol_supply_w_id;
    uint32_t ol_quantity;
};
static_assert(sizeof(OrderLineValue) <= 128);

union ItemKey
{
    using baseType = ChooseBitfieldBaseType<200'000>::type;
    struct
    {
        baseType i_id : ceilLog2(200'000);
    } key;
    baseType base_key = 0;
    ItemKey() = default;
    explicit ItemKey(baseType i_id)
    {
        base_key = 0;
        key.i_id = i_id;
    }
};

struct ItemValue
{
    /* TODO: implement ItemValue */
    uint8_t i_name[24];
    uint32_t i_price;
    uint8_t i_data[52];
    uint32_t i_im_id;
};
static_assert(sizeof(ItemValue) <= 128);

union StockKey
{
    using baseType = ChooseBitfieldBaseType<200'000, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType s_i_id : ceilLog2(200'000);
        baseType s_w_id : ceilLog2(2 * kMaxWarehouses);
    } key;
    baseType base_key = 0;
    StockKey() = default;
    StockKey(baseType s_i_id, baseType s_w_id)
    {
        base_key = 0;
        key.s_i_id = s_i_id;
        key.s_w_id = s_w_id;
    }
};

struct StockValue
{
    uint32_t s_quantity;
    uint32_t s_ytd;
    uint32_t s_order_cnt;
    uint32_t s_remote_cnt;
};
static_assert(sizeof(StockValue) <= 128);

struct StockDataValue
{
    uint8_t s_data[52];
    uint8_t s_dist[10][24];
};

union ClientOrderKey
{
    using baseType = typename ChooseBitfieldBaseType<10'000'000, 20, 2 * kMaxWarehouses, 96'000>::type;
    struct
    {
        baseType oc_o_id : ceilLog2(10'000'000);
        baseType oc_c_id : ceilLog2(96'000);
        baseType oc_d_id : ceilLog2(20);
        baseType oc_w_id : ceilLog2(2 * kMaxWarehouses);
    };
    baseType base_key = 0;
    ClientOrderKey() = default;
    constexpr ClientOrderKey(baseType oc_o_id, baseType oc_c_id, baseType oc_d_id, baseType oc_w_id)
    {
        base_key = 0;
        this->oc_o_id = oc_o_id;
        this->oc_c_id = oc_c_id;
        this->oc_d_id = oc_d_id;
        this->oc_w_id = oc_w_id;
    }
};

} // namespace epic::tpcc

#endif // TPCC_TABLE_H
