//
// Created by Shujian Qian on 2023-09-18.
//

#include <vector>

#include "tpcc_index.h"
#include "tpcc_table.h"
#include "tpcc_txn.h"
#include "tpcc.h"

#include "hashtable_index.h"
#include "util_log.h"

namespace epic::tpcc {

TpccCpuIndex::TpccCpuIndex(const TpccConfig &tpcc_config)
    : tpcc_config(tpcc_config)
    , g_warehouse_index(new StdUnorderedMapIndex<WarehouseKey::baseType, uint32_t>(tpcc_config.warehouseTableSize()))
    , g_district_index(new StdUnorderedMapIndex<DistrictKey::baseType, uint32_t>(tpcc_config.districtTableSize()))
    , g_customer_index(new StdUnorderedMapIndex<CustomerKey::baseType, uint32_t>(tpcc_config.customerTableSize()))
    , g_history_index(new StdUnorderedMapIndex<HistoryKey::baseType, uint32_t>(tpcc_config.historyTableSize()))
    , g_new_order_index(new StdUnorderedMapIndex<NewOrderKey::baseType, uint32_t>(tpcc_config.newOrderTableSize()))
    , g_order_index(new StdUnorderedMapIndex<OrderKey::baseType, uint32_t>(tpcc_config.orderTableSize()))
    , g_order_line_index(new StdUnorderedMapIndex<OrderLineKey::baseType, uint32_t>(tpcc_config.orderLineTableSize()))
    //    , g_history_index(new StdMapIndex<HistoryKey::baseType, uint32_t>(tpcc_config.historyTableSize()))
    //    , g_new_order_index(new StdMapIndex<NewOrderKey::baseType, uint32_t>(tpcc_config.newOrderTableSize()))
    //    , g_order_index(new StdMapIndex<OrderKey::baseType, uint32_t>(tpcc_config.orderTableSize()))
    //    , g_order_line_index(new StdMapIndex<OrderLineKey::baseType, uint32_t>(tpcc_config.orderLineTableSize()))
    , g_item_index(new StdUnorderedMapIndex<ItemKey::baseType, uint32_t>(tpcc_config.itemTableSize()))
    , g_stock_index(new StdUnorderedMapIndex<StockKey::baseType, uint32_t>(tpcc_config.stockTableSize()))

{}

void TpccCpuIndex::indexTxnWrites(BaseTxn *txn, BaseTxn *index, uint32_t epoch_id)
{
    index->txn_type = txn->txn_type;
    switch (static_cast<TpccTxnType>(txn->txn_type))
    {
    case TpccTxnType::NEW_ORDER:
        indexTxnWrites(reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(txn->data),
            reinterpret_cast<NewOrderTxnParams<FixedSizeTxn> *>(index->data), epoch_id);
        break;
    case TpccTxnType::PAYMENT:
        indexTxnWrites(reinterpret_cast<PaymentTxnInput *>(txn->data),
            reinterpret_cast<PaymentTxnParams *>(index->data), epoch_id);
    default:
        /* TODO: implement write indexing for other txn types */
        break;
    }
}

void TpccCpuIndex::indexTxnReads(BaseTxn *txn, BaseTxn *index, uint32_t epoch_id)
{
    index->txn_type = txn->txn_type;
    switch (static_cast<TpccTxnType>(txn->txn_type))
    {
    case TpccTxnType::NEW_ORDER:
        indexTxnReads(reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(txn->data),
            reinterpret_cast<NewOrderTxnParams<FixedSizeTxn> *>(index->data), epoch_id);
        break;
    case TpccTxnType::PAYMENT:
        indexTxnReads(reinterpret_cast<PaymentTxnInput *>(txn->data), reinterpret_cast<PaymentTxnParams *>(index->data),
            epoch_id);
    default:
        /* TODO: implement read indexing for other txn types */
        break;
    }
}

/* Txns' writes need to be indexed before reads */
void TpccCpuIndex::indexTxnWrites(NewOrderTxnInput<FixedSizeTxn> *txn, void *index_ptr, uint32_t epoch_id)
{
    auto index = static_cast<NewOrderTxnParams<FixedSizeTxn> *>(index_ptr);
    OrderKey order_key = {txn->o_id, txn->d_id, txn->origin_w_id};
    index->order_id = g_order_index->searchOrInsert(order_key.base_key);
    NewOrderKey new_order_key = {txn->o_id, txn->d_id, txn->origin_w_id};
    index->new_order_id = g_new_order_index->searchOrInsert(new_order_key.base_key);
    index->num_items = txn->num_items;
    index->all_local = true;
    for (uint32_t i = 0; i < txn->num_items; ++i)
    {
        StockKey stock_key = {txn->items[i].i_id, txn->items[i].w_id};
        index->items[i].stock_id = g_stock_index->searchOrInsert(stock_key.base_key);
        if (txn->items[i].w_id != txn->origin_w_id)
        {
            index->all_local = false;
        }
        OrderLineKey orderline_key = {txn->o_id, txn->d_id, txn->items[i].w_id, i + 1};
        index->items[i].order_line_id = g_order_line_index->searchOrInsert(orderline_key.base_key);
        index->items[i].order_quantities = txn->items[i].order_quantities;
    }
}

void TpccCpuIndex::indexTxnReads(NewOrderTxnInput<FixedSizeTxn> *txn, void *index_ptr, uint32_t epoch_id)
{
    auto index = static_cast<NewOrderTxnParams<FixedSizeTxn> *>(index_ptr);
    CustomerKey customer_key = {txn->c_id, txn->d_id, txn->origin_w_id};
    index->customer_id = g_customer_index->search(customer_key.base_key);
    DistrictKey district_key = {txn->d_id, txn->origin_w_id};
    index->district_id = g_district_index->search(district_key.base_key);
    WarehouseKey warehouse_key{txn->origin_w_id};
    index->warehouse_id = g_warehouse_index->search(warehouse_key.base_key);
    for (uint32_t i = 0; i < txn->num_items; ++i)
    {
        ItemKey item_key{txn->items[i].i_id};
        index->items[i].item_id = g_item_index->search(item_key.base_key);
    }
}
void TpccCpuIndex::indexTxnWrites(PaymentTxnInput *txn, void *index_ptr, uint32_t epoch_id)
{
    auto index = static_cast<PaymentTxnParams *>(index_ptr);
    WarehouseKey warehouse_key{txn->warehouse_id};
    index->warehouse_id = g_warehouse_index->searchOrInsert(warehouse_key.base_key);
    DistrictKey district_key{txn->district_id, txn->warehouse_id};
    index->district_id = g_district_index->search(district_key.base_key);
    CustomerKey customer_key{txn->customer_id, txn->customer_district_id, txn->customer_warehouse_id};
    index->customer_id = g_customer_index->searchOrInsert(customer_key.base_key);
    index->payment_amount = txn->payment_amount;
}

void TpccCpuIndex::indexTxnReads(PaymentTxnInput *txn, void *index_ptr, uint32_t epoch_id) {}

void TpccCpuIndex::loadInitialData()
{
    auto &logger = Logger::GetInstance();
    logger.Info("Loading initial data");

    logger.Trace("Loading warehouse table");
    loadWarehouseTable();

    logger.Trace("Loading district table");
    loadDistrictTable();

    logger.Trace("Loading customer table");
    loadCustomerTable();

    logger.Trace("Loading history table");
    loadHistoryTable();

    logger.Trace("Loading order tables");
    loadOrderTables();

    logger.Trace("Loading item table");
    loadItemTable();

    logger.Trace("Loading stock table");
    loadStockTable();
}

void TpccCpuIndex::loadWarehouseTable()
{
    std::vector<uint32_t> warehouse_ids(tpcc_config.num_warehouses);
    for (uint32_t i = 0; i < tpcc_config.num_warehouses; ++i)
    {
        warehouse_ids[i] = g_warehouse_index->searchOrInsert(WarehouseKey{i + 1}.base_key);
    }
    /* TODO: populate data in Warehouse Table */
}

void TpccCpuIndex::loadDistrictTable()
{
    std::vector<uint32_t> district_ids(tpcc_config.num_warehouses * 10);
    size_t i = 0;
    for (uint32_t w_id = 1; w_id <= tpcc_config.num_warehouses; ++w_id)
    {
        for (uint32_t d_id = 1; d_id <= 10; ++d_id)
        {
            district_ids[i++] = g_district_index->searchOrInsert(DistrictKey{d_id, w_id}.base_key);
        }
    }
    /* TODO: populate data in District Table */
}

void TpccCpuIndex::loadCustomerTable()
{
    std::vector<uint32_t> customer_ids(tpcc_config.num_warehouses * 10 * 3'000);
    size_t i = 0;
    for (uint32_t w_id = 1; w_id <= tpcc_config.num_warehouses; ++w_id)
    {
        for (uint32_t d_id = 1; d_id <= 10; ++d_id)
        {
            for (uint32_t c_id = 1; c_id <= 3'000; ++c_id)
            {
                customer_ids[i++] = g_customer_index->searchOrInsert(CustomerKey{c_id, d_id, w_id}.base_key);
            }
        }
    }
    /* TODO: populate data in Customer Table */
}

void TpccCpuIndex::loadHistoryTable()
{
    /* TODO: implement loadHistoryTable */
}

void TpccCpuIndex::loadOrderTables()
{
    std::vector<uint32_t> order_ids(tpcc_config.num_warehouses * 10 * 3'000);
    std::vector<uint32_t> new_order_ids(tpcc_config.num_warehouses * 10 * 900);
    std::vector<uint32_t> order_line_ids(tpcc_config.num_warehouses * 10 * 3'000 * 15);
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    for (uint32_t w_id = 1; w_id <= tpcc_config.num_warehouses; ++w_id)
    {
        for (uint32_t d_id = 1; d_id <= 10; ++d_id)
        {
            for (uint32_t o_id = 1; o_id <= 3'000; ++o_id)
            {
                order_ids[i++] = g_order_index->searchOrInsert(OrderKey{o_id, d_id, w_id}.base_key);
                if (o_id > 2'100)
                    new_order_ids[j++] = g_new_order_index->searchOrInsert(NewOrderKey{o_id, d_id, w_id}.base_key);
                for (uint32_t ol_number = 1; ol_number <= 15; ++ol_number)
                {
                    order_line_ids[k++] =
                        g_order_line_index->searchOrInsert(OrderLineKey{o_id, d_id, w_id, ol_number}.base_key);
                }
            }
        }
    }
    /* TODO: populate data in Order Table and New Order Table */
}

void TpccCpuIndex::loadItemTable()
{
    std::vector<uint32_t> item_ids(100'000);
    size_t i = 0;
    for (uint32_t i_id = 1; i_id <= 100'000; ++i_id)
    {
        item_ids[i++] = g_item_index->searchOrInsert(ItemKey{i_id}.base_key);
    }
    /* TODO: populate data in Item Table */
}

void TpccCpuIndex::loadStockTable()
{
    std::vector<uint32_t> stock_ids(tpcc_config.num_warehouses * 100'000);
    size_t i = 0;
    for (uint32_t w_id = 1; w_id <= tpcc_config.num_warehouses; ++w_id)
    {
        for (uint32_t i_id = 1; i_id <= 100'000; ++i_id)
        {
            stock_ids[i++] = g_stock_index->searchOrInsert(StockKey{i_id, w_id}.base_key);
        }
    }
    /* TODO: populate data in Stock Table */
}

void TpccCpuIndex::indexTxns(TxnArray<TpccTxn> &txn_array, TxnArray<TpccTxnParam> &index_array, uint32_t index_epoch_id)
{
    /* it's important to index writes before reads */
    for (uint32_t i = 0; i < tpcc_config.num_txns; ++i)
    {
        BaseTxn *txn = txn_array.getTxn(i);
        BaseTxn *txn_param = index_array.getTxn(i);
        indexTxnWrites(txn, txn_param, index_epoch_id);
    }
    for (uint32_t i = 0; i < tpcc_config.num_txns; ++i)
    {
        BaseTxn *txn = txn_array.getTxn(i);
        BaseTxn *txn_param = index_array.getTxn(i);
        indexTxnReads(txn, txn_param, index_epoch_id);
    }
}

} // namespace epic::tpcc