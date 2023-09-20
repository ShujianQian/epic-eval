//
// Created by Shujian Qian on 2023-08-23.
//

#include "tpcc_txn.h"

namespace epic::tpcc {

void runTransaction(BaseTxn *txn)
{
    switch (static_cast<TpccTxnType>(txn->txn_type))
    {
    case TpccTxnType::NEW_ORDER:
        runTransaction(reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(txn->data));
        break;
    case TpccTxnType::PAYMENT:
        runTransaction(reinterpret_cast<PaymentTxn *>(txn->data));
        break;
    default:
        break;
    }
}

void runTransaction(NewOrderTxnInput<FixedSizeTxn> *txn) {}

void runTransaction(PaymentTxn *txn) {}

} // namespace epic::tpcc