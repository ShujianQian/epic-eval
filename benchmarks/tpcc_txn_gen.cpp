//
// Created by Shujian Qian on 2023-11-05.
//

#include <benchmarks/tpcc_txn_gen.h>

namespace epic::tpcc {

TpccTxnType TpccTxnGenerator::getTxnType()
{
    uint32_t txn_type = txn_type_dist(gen);
    uint32_t acc = 0;
    if (txn_type < acc + config.txn_mix.new_order)
    {
        return TpccTxnType::NEW_ORDER;
    }
    acc += config.txn_mix.new_order;
    if (txn_type < acc + config.txn_mix.payment)
    {
        return TpccTxnType::PAYMENT;
    }
    acc += config.txn_mix.payment;
    if (txn_type < acc + config.txn_mix.order_status)
    {
        return TpccTxnType::ORDER_STATUS;
    }
    acc += config.txn_mix.order_status;
    if (txn_type < acc + config.txn_mix.delivery)
    {
        return TpccTxnType::DELIVERY;
    }
    return TpccTxnType::STOCK_LEVEL;
}

void TpccTxnGenerator::generateTxn(NewOrderTxnInput<FixedSizeTxn> *txn, uint32_t timestamp)
{
    /* TODO: generate rollbacks? */
    txn->origin_w_id = w_id_dist(gen);
    txn->o_id = o_id_gen.getOId(timestamp);
    txn->d_id = d_id_dist(gen);
    txn->c_id = c_id_dist(gen);
    txn->num_items = num_item_dist(gen);
    for (size_t i = 0; i < txn->num_items; ++i)
    {
        /* generate unique item indexes */
        bool retry;
        do
        {
            txn->items[i].i_id = i_id_dist(gen);
            retry = false;
            for (size_t j = 0; j + 1 < i; ++j)
            {
                if (txn->items[i].i_id == txn->items[j].i_id)
                {
                    retry = true;
                    break;
                }
            }
        } while (retry);

        /* no remote warehouse for single warehouse configs, otherwise 1% remote orders */
        bool supply_from_remote = config.num_warehouses > 1 && (percentage_gen(gen) == 1);
        if (supply_from_remote)
        {
            do
            {
                txn->items[i].w_id = w_id_dist(gen);
            } while (txn->items[i].w_id == txn->origin_w_id);
        }
        else
        {
            txn->items[i].w_id = txn->origin_w_id;
        }

        txn->items[i].order_quantities = order_quantity_dist(gen);
    }
}

void TpccTxnGenerator::generateTxn(PaymentTxnInput *txn, uint32_t timestamp)
{
    txn->warehouse_id = w_id_dist(gen);
    txn->district_id = d_id_dist(gen);

    /* no remote payment for singel warehouse configs, otherwise 15% remote payments */
    bool pay_from_remote = config.num_warehouses > 1 && (percentage_gen(gen) <= 15);
    if (pay_from_remote)
    {
        do
        {
            txn->customer_warehouse_id = w_id_dist(gen);
        } while (txn->customer_warehouse_id == txn->warehouse_id);
        txn->customer_district_id = d_id_dist(gen);
    }
    else
    {
        txn->customer_warehouse_id = txn->warehouse_id;
        txn->customer_district_id = txn->district_id;
    }

    txn->customer_id = c_id_dist(gen);
    txn->payment_amount = payment_amount_dist(gen);
}

void TpccTxnGenerator::generateTxn(OrderStatusTxn *txn, uint32_t timestamp)
{
    /* TODO: generate order-status txn */
}

void TpccTxnGenerator::generateTxn(DeliveryTxn *txn, uint32_t timestamp)
{
    /* TODO: generate delivery txn */
}

void TpccTxnGenerator::generateTxn(StockLevelTxn *txn, uint32_t timestamp)
{
    /* TODO: generate stock-level txn */
}

void TpccTxnGenerator::generateTxn(BaseTxn *txn, uint32_t timestamp)
{
    TpccTxnType txn_type = getTxnType();
    txn->txn_type = static_cast<uint32_t>(txn_type);
    switch (txn_type)
    {
    case TpccTxnType::NEW_ORDER:
        generateTxn(reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(txn->data), timestamp);
        break;
    case TpccTxnType::PAYMENT:
        generateTxn(reinterpret_cast<PaymentTxnInput *>(txn->data), timestamp);
        break;
    case TpccTxnType::ORDER_STATUS:
        generateTxn(reinterpret_cast<OrderStatusTxn *>(txn->data), timestamp);
        break;
    case TpccTxnType::DELIVERY:
        generateTxn(reinterpret_cast<DeliveryTxn *>(txn->data), timestamp);
        break;
    case TpccTxnType::STOCK_LEVEL:
        generateTxn(reinterpret_cast<StockLevelTxn *>(txn->data), timestamp);
        break;
    default:
        break;
    }
}

}