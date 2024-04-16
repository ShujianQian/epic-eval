//
// Created by Shujian Qian on 2023-11-20.
//

#ifndef EPIC_BENCHMARKS_TPCC_INDEX_H
#define EPIC_BENCHMARKS_TPCC_INDEX_H

#include <txn.h>
#include <base_index.h>
#include <std_map_index.h>
#include <std_unordered_map_index.h>
#include "tpcc_config.h"
#include <benchmarks/tpcc_table.h>

namespace epic::tpcc {

template <typename TxnArrayType, typename TxnParamArrayType>
class TpccIndex
{
public:
    virtual void indexTxns(TxnArrayType &txn_array, TxnParamArrayType &index_array, uint32_t epoch_id) = 0;
    virtual void loadInitialData() = 0;
    virtual ~TpccIndex() = default;
};
//
// class TpccCpuIndex : public TpccIndex
//{
// private:
//    std::unique_ptr<UnorderedIndex<WarehouseKey>> g_warehouse_index;
//    std::unique_ptr<UnorderedIndex<DistrictKey>> g_district_index;
//    std::unique_ptr<UnorderedIndex<CustomerKey>> g_customer_index;
//    std::unique_ptr<UnorderedIndex<HistoryKey>> g_history_index;
//    std::unique_ptr<UnorderedIndex<NewOrderKey>> g_new_order_index;
//    std::unique_ptr<UnorderedIndex<OrderKey>> g_order_index;
//    std::unique_ptr<UnorderedIndex<OrderLineKey>> g_order_line_index;
//    std::unique_ptr<UnorderedIndex<ItemKey>> g_item_index;
//    std::unique_ptr<UnorderedIndex<StockKey>> g_stock_index;
//
//    const TpccConfig tpcc_config;
//
// public:
//    TpccCpuIndex(const TpccConfig &tpcc_config);
//
//    void indexTxnWrites(BaseTxn *txn, BaseTxn *index, uint32_t epoch_id);
//    void indexTxnWrites(NewOrderTxnInput<FixedSizeTxn> *txn, void *index_ptr, uint32_t epoch_id);
//    void indexTxnWrites(PaymentTxnInput *txn, void *index_ptr, uint32_t epoch_id);
//
//    void indexTxnReads(BaseTxn *txn, BaseTxn *index, uint32_t epoch_id);
//    void indexTxnReads(NewOrderTxnInput<FixedSizeTxn> *txn, void *index_ptr, uint32_t epoch_id);
//    void indexTxnReads(PaymentTxnInput *txn, void *index_ptr, uint32_t epoch_id);
//
//    void loadWarehouseTable();
//    void loadDistrictTable();
//    void loadCustomerTable();
//    void loadHistoryTable();
//    void loadOrderTables();
//    void loadItemTable();
//    void loadStockTable();
//
//    void loadInitialData() override;
//    void indexTxns(TxnArray<TpccTxn> &txn_array, TxnArray<TpccTxnParam> &index_array, uint32_t epoch_id) override;
//
//    friend class TpccLoader;
//};

template <typename TxnArrayType, typename TxnParamArrayType>
class TpccCpuIndex : public TpccIndex<TxnArrayType, TxnParamArrayType>
{
private:
    std::unique_ptr<BaseIndex<WarehouseKey::baseType, uint32_t>> g_warehouse_index;
    std::unique_ptr<BaseIndex<DistrictKey::baseType, uint32_t>> g_district_index;
    std::unique_ptr<BaseIndex<CustomerKey::baseType, uint32_t>> g_customer_index;
    std::unique_ptr<BaseIndex<HistoryKey::baseType, uint32_t>> g_history_index;
    std::unique_ptr<BaseIndex<NewOrderKey::baseType, uint32_t>> g_new_order_index;
    std::unique_ptr<BaseIndex<OrderKey::baseType, uint32_t>> g_order_index;
    std::unique_ptr<BaseIndex<OrderLineKey::baseType, uint32_t>> g_order_line_index;
    std::unique_ptr<BaseIndex<ItemKey::baseType, uint32_t>> g_item_index;
    std::unique_ptr<BaseIndex<StockKey::baseType, uint32_t>> g_stock_index;

    const TpccConfig tpcc_config;

public:
    TpccCpuIndex(const TpccConfig &tpcc_config);

    void indexTxnWrites(BaseTxn *txn, BaseTxn *index, uint32_t epoch_id);
    void indexTxnWrites(NewOrderTxnInput<FixedSizeTxn> *txn, void *index_ptr, uint32_t epoch_id);
    void indexTxnWrites(PaymentTxnInput *txn, void *index_ptr, uint32_t epoch_id);

    void indexTxnReads(BaseTxn *txn, BaseTxn *index, uint32_t epoch_id);
    void indexTxnReads(NewOrderTxnInput<FixedSizeTxn> *txn, void *index_ptr, uint32_t epoch_id);
    void indexTxnReads(PaymentTxnInput *txn, void *index_ptr, uint32_t epoch_id);

    void loadWarehouseTable();
    void loadDistrictTable();
    void loadCustomerTable();
    void loadHistoryTable();
    void loadOrderTables();
    void loadItemTable();
    void loadStockTable();

    void loadInitialData() override;
    void indexTxns(TxnArrayType &txn_array, TxnParamArrayType &index_array, uint32_t epoch_id) override;

    friend class TpccLoader;
};

// class TpccLoader
//{
// private:
//     const TpccConfig &tpcc_config;
//     const TpccIndex &index;
//
//
// public:
//     TpccLoader(const TpccConfig &tpcc_config, const TpccIndex &index)
//         : tpcc_config(tpcc_config)
//         , index(index){};
//
//     void loadInitialData();
// };

} // namespace epic::tpcc

#endif // EPIC_BENCHMARKS_TPCC_INDEX_H
