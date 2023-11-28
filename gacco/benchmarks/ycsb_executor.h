//
// Created by Shujian Qian on 2023-11-27.
//

#ifndef EPIC_GACCO_BENCHMARKS_YCSB_EXECUTOR_H
#define EPIC_GACCO_BENCHMARKS_YCSB_EXECUTOR_H

#include <gacco/execution_planner.h>
#include <gacco/benchmarks/ycsb_txn.h>
#include <benchmarks/ycsb_config.h>
#include <benchmarks/ycsb_table.h>
#include <txn.h>

namespace gacco::ycsb {

using epic::TxnArray;
using epic::ycsb::YcsbConfig;
using epic::ycsb::YcsbValue;

class Executor
{
public:
    GaccoTableLock lock;
    YcsbConfig config;
    TxnArray<YcsbTxnParam> txn;
    YcsbValue *records;

    Executor(YcsbValue *records, GaccoTableLock lock, TxnArray<YcsbTxnParam> txn, YcsbConfig config)
        : lock(lock)
        , records(records)
        , txn(txn)
        , config(config){};
    virtual ~Executor() = default;
    virtual void execute(uint32_t epoch) = 0;
};

} // namespace gacco::ycsb

#endif // EPIC_GACCO_BENCHMARKS_YCSB_EXECUTOR_H
