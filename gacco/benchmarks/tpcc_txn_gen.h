//
// Created by Shujian Qian on 2023-11-05.
//

#ifndef EPIC_GACCO_BENCHMARKS_TPCC_TXN_GEN_H
#define EPIC_GACCO_BENCHMARKS_TPCC_TXN_GEN_H

#include <benchmarks/tpcc_txn_gen.h>
#include <gacco/benchmarks/tpcc_txn.h>

namespace gacco::tpcc {

struct TpccTxnGenerator : public epic::tpcc::TpccTxnGenerator
{
    explicit TpccTxnGenerator (epic::tpcc::TpccConfig config) : epic::tpcc::TpccTxnGenerator(config) {}
    TpccTxnType getTxnType(uint32_t txn_id);
    void generateTxn(epic::BaseTxn *base_txn, uint32_t txn_id, uint32_t timestamp);
};

}

#endif // EPIC_GACCO_BENCHMARKS_TPCC_TXN_GEN_H
