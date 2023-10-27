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

struct WarehouseValue
{
    /* TODO: implement WarehouseValue */
    uint32_t placeholder;
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

struct DistrictValue
{
    /* TODO: implement DistrictValue */
    uint32_t placeholder;
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

struct CustomerValue
{
    /* TODO: implement CustomerValue */
    uint32_t placeholder;
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
    baseType base_key;
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
    /* TODO: implement HistoryValue */
    uint32_t placeholder;
};

union NewOrderKey
{
    using baseType = ChooseBitfieldBaseType<10'000'000, 20, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType no_o_id : ceilLog2(10'000'000);
        baseType no_d_id : ceilLog2(20);
        baseType no_w_id : ceilLog2(2 * kMaxWarehouses);
    };
    baseType base_key;
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
    /* TODO: implement NewOrderValue */
    uint32_t placeholder;
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
    baseType base_key;
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
    /* TODO: implement OrderValue */
    uint32_t o_c_id;
    uint32_t placeholder;
};

union OrderLineKey
{
    using baseType = ChooseBitfieldBaseType<10'000'000, 20, 2 * kMaxWarehouses, 15>::type;
    struct
    {
        baseType ol_o_id : ceilLog2(10'000'000);
        baseType ol_d_id : ceilLog2(20);
        baseType ol_w_id : ceilLog2(2 * kMaxWarehouses);
        baseType ol_number : ceilLog2(15);
    } key;
    baseType base_key;
    OrderLineKey(baseType ol_o_id, baseType ol_d_id, baseType ol_w_id, baseType ol_number)
    {
        base_key = 0;
        key.ol_o_id = ol_o_id;
        key.ol_d_id = ol_d_id;
        key.ol_w_id = ol_w_id;
        key.ol_number = ol_number;
    }
};

struct OrderLineValue
{
    /* TODO: implement OrderLineValue */
    uint32_t placeholder;
};

union ItemKey
{
    using baseType = ChooseBitfieldBaseType<200'000>::type;
    struct
    {
        baseType i_id : ceilLog2(200'000);
    } key;
    baseType base_key;
    explicit ItemKey(baseType i_id)
    {
        base_key = 0;
        key.i_id = i_id;
    }
};

struct ItemValue
{
    /* TODO: implement ItemValue */
    uint32_t placeholder;
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

struct StockValue
{
    /* TODO: implement StockValue */
    uint32_t placeholder;
};

class TpccIndex
{
private:
    std::unique_ptr<UnorderedIndex<WarehouseKey>> g_warehouse_index;
    std::unique_ptr<UnorderedIndex<DistrictKey>> g_district_index;
    std::unique_ptr<UnorderedIndex<CustomerKey>> g_customer_index;
    std::unique_ptr<UnorderedIndex<HistoryKey>> g_history_index;
    std::unique_ptr<UnorderedIndex<NewOrderKey>> g_new_order_index;
    std::unique_ptr<UnorderedIndex<OrderKey>> g_order_index;
    std::unique_ptr<UnorderedIndex<OrderLineKey>> g_order_line_index;
    std::unique_ptr<UnorderedIndex<ItemKey>> g_item_index;
    std::unique_ptr<UnorderedIndex<StockKey>> g_stock_index;

    const TpccConfig &tpcc_config;

public:
    TpccIndex(const TpccConfig &tpcc_config);

    void indexTxnWrites(BaseTxn *txn, BaseTxn *index, uint32_t epoch_id);
    void indexTxnWrites(NewOrderTxnInput<FixedSizeTxn> *txn, void *index_ptr, uint32_t epoch_id);

    void indexTxnReads(BaseTxn *txn, BaseTxn *index, uint32_t epoch_id);
    void indexTxnReads(NewOrderTxnInput<FixedSizeTxn> *txn, void *index_ptr, uint32_t epoch_id);

    friend class TpccLoader;
};

class TpccLoader
{
private:
    const TpccConfig &tpcc_config;
    const TpccIndex &index;

    void loadWarehouseTable();
    void loadDistrictTable();
    void loadCustomerTable();
    void loadHistoryTable();
    void loadOrderTables();
    void loadItemTable();
    void loadStockTable();

public:
    TpccLoader(const TpccConfig &tpcc_config, const TpccIndex &index)
        : tpcc_config(tpcc_config)
        , index(index){};

    void loadInitialData();
};

} // namespace epic::tpcc

#endif // TPCC_TABLE_H