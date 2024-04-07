//
// Created by Shujian Qian on 2023-11-05.
//

#ifndef EPIC_GACCO_BENCHMARKS_TPCC_TXN_H
#define EPIC_GACCO_BENCHMARKS_TPCC_TXN_H

#include <benchmarks/tpcc_txn.h>

namespace gacco::tpcc {

using epic::tpcc::TpccTxnType;

using epic::tpcc::FixedSizeTxn;

using epic::tpcc::NewOrderTxnInput;
using epic::tpcc::PaymentTxnInput;
using epic::tpcc::OrderStatusTxnInput;
using epic::tpcc::DeliveryTxn;
using epic::tpcc::StockLevelTxn;

using epic::tpcc::TpccTxnParam;
using epic::tpcc::NewOrderTxnParams;
using epic::tpcc::PaymentTxnParams;

}

#endif // EPIC_GACCO_BENCHMARKS_TPCC_TXN_H
