//
// Created by Shujian Qian on 2023-12-04.
//

#include <benchmarks/tpcc_cpu_executor.h>

#include <thread>

#include <storage.h>

namespace epic::tpcc {

template<typename TxnParamArrayType, typename TxnExecPlanArrayType>
void CpuExecutor<TxnParamArrayType, TxnExecPlanArrayType>::execute(uint32_t epoch)
{
    uint32_t num_threads = config.cpu_exec_num_threads;
    std::vector<std::thread> threads;
    for (uint32_t i = 0; i < num_threads; i++)
    {
        threads.emplace_back(&CpuExecutor::executionWorker, this, epoch, i);
    }

    for (auto &t : threads)
    {
        t.join();
    }
}

template<typename TxnParamArrayType, typename TxnExecPlanArrayType>
inline void CpuExecutor<TxnParamArrayType, TxnExecPlanArrayType>::executeTxn(
    NewOrderTxnParams<FixedSizeTxn> *txn, NewOrderExecPlan<FixedSizeTxn> *plan, uint32_t epoch)
{
    double mult = 1;
    WarehouseValue w;
    readFromTable(
        records.warehouse_record, versions.warehouse_version, txn->warehouse_id, plan->warehouse_loc, epoch, &w);
    mult += w.w_tax / 100.0;
    DistrictValue d;
    readFromTable(records.district_record, versions.district_version, txn->district_id, plan->district_loc, epoch, &d);
    d.d_next_o_id = txn->next_order_id;
    writeToTable(
        records.district_record, versions.district_version, txn->district_id, plan->district_write_loc, epoch, &d);
    mult += d.d_tax / 100.0;
    CustomerValue c;
    readFromTable(records.customer_record, versions.customer_version, txn->customer_id, plan->customer_loc, epoch, &c);
    mult *= 1 - c.c_discount / 100.0;
    OrderValue o;
    o.o_all_local = txn->all_local;
    o.o_ol_cnt = txn->num_items;
    writeToTable(records.order_record, versions.order_version, txn->order_id, plan->order_loc, epoch, &o);
    NewOrderValue no;
    no.dummy = epoch;
    writeToTable(
        records.new_order_record, versions.new_order_version, txn->new_order_id, plan->new_order_loc, epoch, &no);
    double total = 0;
    for (uint32_t i = 0; i < txn->num_items; i++)
    {
        ItemValue it;
        readFromTable(records.item_record, versions.item_version, txn->items[i].item_id, plan->item_plans[i].item_loc,
            epoch, &it);
        total += it.i_price;
        OrderLineValue ol;
        ol.ol_amount = txn->items[i].order_quantities;
        writeToTable(records.order_line_record, versions.order_line_version, txn->items[i].order_line_id,
            plan->item_plans[i].orderline_loc, epoch, &ol);
        StockValue s;
        readFromTable(records.stock_record, versions.stock_version, txn->items[i].stock_id,
            plan->item_plans[i].stock_read_loc, epoch, &s);
        s.s_quantity = (s.s_quantity >= txn->items[i].order_quantities + 10)
                           ? s.s_quantity - txn->items[i].order_quantities
                           : s.s_quantity + 91 - txn->items[i].order_quantities;
        writeToTable(records.stock_record, versions.stock_version, txn->items[i].stock_id,
            plan->item_plans[i].stock_write_loc, epoch, &s);
    }
    total *= mult;
    txn->num_items = total; /* so that the compiler won't optimize the code above away */
}

template<typename TxnParamArrayType, typename TxnExecPlanArrayType>
inline void CpuExecutor<TxnParamArrayType, TxnExecPlanArrayType>::executeTxn(
    PaymentTxnParams *txn, PaymentTxnExecPlan *plan, uint32_t epoch)
{
    WarehouseValue w;
    readFromTable(
        records.warehouse_record, versions.warehouse_version, txn->warehouse_id, plan->warehouse_read_loc, epoch, &w);
    w.w_ytd += txn->payment_amount;
    writeToTable(
        records.warehouse_record, versions.warehouse_version, txn->warehouse_id, plan->warehouse_write_loc, epoch, &w);
    DistrictValue d;
    readFromTable(
        records.district_record, versions.district_version, txn->district_id, plan->district_read_loc, epoch, &d);
    d.d_ytd += txn->payment_amount;
    writeToTable(
        records.district_record, versions.district_version, txn->district_id, plan->district_write_loc, epoch, &d);
    CustomerValue c;
    readFromTable(
        records.customer_record, versions.customer_version, txn->customer_id, plan->customer_read_loc, epoch, &c);
    c.c_balance -= txn->payment_amount;
    c.c_ytd_payment += txn->payment_amount;
    c.c_payment_cnt++;
    writeToTable(
        records.customer_record, versions.customer_version, txn->customer_id, plan->customer_write_loc, epoch, &c);
}

template<typename TxnParamArrayType, typename TxnExecPlanArrayType>
void CpuExecutor<TxnParamArrayType, TxnExecPlanArrayType>::executionWorker(uint32_t epoch, uint32_t tid)
{
    uint32_t num_threads = config.cpu_exec_num_threads;
    for (uint32_t txn_id = tid; txn_id < config.num_txns;
         txn_id += num_threads) /* evenly partition transactions to threads */
    {
        BaseTxn *base_txn = txn.getTxn(txn_id);
        BaseTxn *base_plan = plan.getTxn(txn_id);
        TpccTxnType txn_type = static_cast<TpccTxnType>(base_txn->txn_type);
        switch (txn_type)
        {
        case TpccTxnType::NEW_ORDER:
            executeTxn(reinterpret_cast<NewOrderTxnParams<FixedSizeTxn> *>(base_txn->data),
                reinterpret_cast<NewOrderExecPlan<FixedSizeTxn> *>(base_plan->data), epoch);
            break;
        case TpccTxnType::PAYMENT:
            executeTxn(reinterpret_cast<PaymentTxnParams *>(base_txn->data),
                reinterpret_cast<PaymentTxnExecPlan *>(base_plan->data), epoch);
            break;
        case TpccTxnType::ORDER_STATUS:
            executeTxn(reinterpret_cast<OrderStatusTxnParams *>(base_txn->data),
                reinterpret_cast<OrderStatusTxnExecPlan *>(base_plan->data), epoch);
            break;
        case TpccTxnType::DELIVERY:
            executeTxn(reinterpret_cast<DeliveryTxnParams *>(base_txn->data),
                reinterpret_cast<DeliveryTxnExecPlan *>(base_plan->data), epoch);
            break;
        case TpccTxnType::STOCK_LEVEL:
            executeTxn(reinterpret_cast<StockLevelTxnParams *>(base_txn->data),
                reinterpret_cast<StockLevelTxnExecPlan *>(base_plan->data), epoch);
            break;
        default:
            throw std::runtime_error("Unsupported transaction type.");
        }
    }
}

template<typename TxnParamArrayType, typename TxnExecPlanArrayType>
void CpuExecutor<TxnParamArrayType, TxnExecPlanArrayType>::executeTxn(
    OrderStatusTxnParams *txn, OrderStatusTxnExecPlan *plan, uint32_t epoch)
{
    CustomerValue cv;
    readFromTable(records.customer_record, versions.customer_version, txn->customer_id, plan->customer_loc, epoch, &cv);
    OrderValue ov;
    readFromTable(records.order_record, versions.order_version, txn->order_id, plan->order_loc, epoch, &ov);
    for (int i = 0; i < txn->num_items; ++i)
    {
        OrderLineValue olv;
        readFromTable(records.order_line_record, versions.order_line_version, txn->orderline_ids[i],
            plan->orderline_locs[i], epoch, &olv);
    }
}

template<typename TxnParamArrayType, typename TxnExecPlanArrayType>
void CpuExecutor<TxnParamArrayType, TxnExecPlanArrayType>::executeTxn(
    DeliveryTxnParams *txn, DeliveryTxnExecPlan *plan, uint32_t epoch)
{

    for (int i = 0; i < 10; ++i)
    {
        NewOrderValue nov;
        readFromTable(records.new_order_record, versions.new_order_version, txn->new_order_id[i],
            plan->new_order_read_locs[i], epoch, &nov);

        OrderValue ov;
        readFromTable(
            records.order_record, versions.order_version, txn->order_id[i], plan->order_read_locs[i], epoch, &ov);
        ov.o_carrier_id = txn->carrier_id;
        writeToTable(
            records.order_record, versions.order_version, txn->order_id[i], plan->order_write_locs[i], epoch, &ov);

        uint32_t amount = 0;
        for (int j = 0; j < txn->num_items[i]; ++j)
        {
            OrderLineValue olv;
            readFromTable(records.order_line_record, versions.order_line_version, txn->orderline_ids[i][j],
                plan->orderline_read_locs[i][j], epoch, &olv);
            amount += olv.ol_amount;
            olv.ol_delivery_d = txn->delivery_d;
            writeToTable(records.order_line_record, versions.order_line_version, txn->orderline_ids[i][j],
                plan->orderline_write_locs[i][j], epoch, &olv);
        }

        CustomerValue cv;
        readFromTable(records.customer_record, versions.customer_version, txn->customer_id[i],
            plan->customer_read_locs[i], epoch, &cv);
        cv.c_balance += amount;
        ++cv.c_delivery_cnt;
        writeToTable(records.customer_record, versions.customer_version, txn->customer_id[i],
            plan->customer_write_locs[i], epoch, &cv);
    }
}

template<typename TxnParamArrayType, typename TxnExecPlanArrayType>
void CpuExecutor<TxnParamArrayType, TxnExecPlanArrayType>::executeTxn(
    StockLevelTxnParams *txn, StockLevelTxnExecPlan *plan, uint32_t epoch)
{
    uint32_t num_low_stock = 0;
    const uint32_t threshold = txn->threshold;
    StockValue sv;
    for (int i = 0; i < txn->num_items; ++i)
    {
        readFromTable(
            records.stock_record, versions.stock_version, txn->stock_ids[i], plan->stock_read_locs[i], epoch, &sv);
        if (sv.s_quantity < threshold)
        {
            ++num_low_stock;
        }
    }
    txn->num_low_stock = num_low_stock;
}

template class CpuExecutor<TpccTxnParamArrayT, TpccTxnExecPlanArrayT>;

} // namespace epic::tpcc