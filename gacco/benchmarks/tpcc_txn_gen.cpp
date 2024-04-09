//
// Created by Shujian Qian on 2023-11-05.
//

#include <gacco/benchmarks/tpcc_txn_gen.h>
#include <gacco/benchmarks/tpcc_txn.h>

#include <util_log.h>

namespace gacco::tpcc {

void TpccTxnGenerator::generateTxn(epic::BaseTxn *txn, uint32_t txn_id, uint32_t timestamp) {

    TpccTxnType txn_type = getTxnType(txn_id);
    txn->txn_type = static_cast<uint32_t>(txn_type);
    switch (txn_type)
    {
    case TpccTxnType::NEW_ORDER:
        epic::tpcc::TpccTxnGenerator::generateTxn(reinterpret_cast<epic::tpcc::NewOrderTxnInput<FixedSizeTxn> *>(txn->data), timestamp);
        break;
    case TpccTxnType::PAYMENT:
        epic::tpcc::TpccTxnGenerator::generateTxn(reinterpret_cast<PaymentTxnInput *>(txn->data), timestamp);
        break;
    case TpccTxnType::ORDER_STATUS:
        epic::tpcc::TpccTxnGenerator::generateTxn(reinterpret_cast<OrderStatusTxnInput *>(txn->data), timestamp);
        break;
    case TpccTxnType::DELIVERY:
        epic::tpcc::TpccTxnGenerator::generateTxn(reinterpret_cast<DeliveryTxnInput *>(txn->data), timestamp);
        break;
    case TpccTxnType::STOCK_LEVEL:
        epic::tpcc::TpccTxnGenerator::generateTxn(reinterpret_cast<StockLevelTxnInput *>(txn->data), timestamp);
        break;
    default:
        break;
    }
}

TpccTxnType TpccTxnGenerator::getTxnType(uint32_t txn_id)
{
    auto &logger = epic::Logger::GetInstance();
    if (config.gacco_separate_txn_queue) {
        if (txn_id < config.txn_mix.new_order * config.num_txns / 100) {
            return TpccTxnType::NEW_ORDER;
        }
        /* gacco only runs new order and payment on GPU */
        return TpccTxnType::PAYMENT;
    } else {
        TpccTxnType retval;
        do {
            retval = epic::tpcc::TpccTxnGenerator::getTxnType();
        } while (!(retval == TpccTxnType::NEW_ORDER || retval == TpccTxnType::PAYMENT));
        /* gacco only runs new order and payment on GPU */
        return retval;
    }
}

}