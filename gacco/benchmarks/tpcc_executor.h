//
// Created by Shujian Qian on 2023-11-10.
//

#ifndef EPIC_GACCO_BENCHMARKS_TPCC_EXECUTOR_H
#define EPIC_GACCO_BENCHMARKS_TPCC_EXECUTOR_H

#include <gacco/execution_planner.h>
#include <gacco/benchmarks/tpcc_txn.h>
#include <gacco/benchmarks/tpcc_storage.h>
#include "benchmarks/tpcc_config.h"
#include <txn.h>

namespace gacco::tpcc {

using epic::TxnArray;

class Executor
{
public:
    struct TpccTableLocks
    {
        GaccoTableLock warehouse, district, customer, history, new_order, order, order_line, item, stock;
    } table_locks;
    epic::tpcc::TpccConfig config;
    TxnArray<TpccTxnParam> txn;
    TpccRecords records;

    Executor(TpccRecords records, TpccTableLocks table_locks, TxnArray<TpccTxnParam> txn, epic::tpcc::TpccConfig config)
        : table_locks(table_locks)
        , records(records)
        , txn(txn)
        , config(config){};
    virtual ~Executor() = default;

    virtual void execute(uint32_t epoch) = 0;
};

} // namespace gacco::tpcc

#endif // EPIC_GACCO_BENCHMARKS_TPCC_EXECUTOR_H
