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
};

struct WarehouseValue
{
    /* TODO: implement WarehouseValue */
};

struct DistrictKey
{
    using baseType = ChooseBitfieldBaseType<20, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType d_id : ceilLog2(20);
        baseType d_w_id : ceilLog2(2 * kMaxWarehouses);
    } key;
    baseType base_key;
};

struct CustomerKey
{
    using baseType = ChooseBitfieldBaseType<96'000, 20, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType c_id : ceilLog2(96'000);
        baseType c_d_id : ceilLog2(20);
        baseType c_w_id : ceilLog2(2 * kMaxWarehouses);
    } key;
    baseType base_key;
};

struct HistoryKey
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
};

struct NewOrderKey
{
    using baseType = ChooseBitfieldBaseType<10'000'000, 20, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType no_o_id : ceilLog2(10'000'000);
        baseType no_d_id : ceilLog2(20);
        baseType no_w_id : ceilLog2(2 * kMaxWarehouses);
    };
    baseType base_key;
};

struct OrderKey
{
    using baseType = typename ChooseBitfieldBaseType<10'000'000, 20, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType o_id : ceilLog2(10'000'000);
        baseType o_d_id : ceilLog2(20);
        baseType o_w_id : ceilLog2(2 * kMaxWarehouses);
    };
    baseType base_key;
};

struct OrderValue
{
    uint32_t o_c_id;
};

struct OrderLineKey
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
};

struct ItemKey
{
    using baseType = ChooseBitfieldBaseType<200'000>::type;
    struct
    {
        baseType i_id : ceilLog2(200'000);
    } key;
    baseType base_key;
};

struct StockKey
{
    using baseType = ChooseBitfieldBaseType<200'000, 2 * kMaxWarehouses>::type;
    struct
    {
        baseType s_i_id : ceilLog2(200'000);
        baseType s_w_id : ceilLog2(2 * kMaxWarehouses);
    } key;
    baseType base_key;
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
