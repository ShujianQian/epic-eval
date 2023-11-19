//
// Created by Shujian Qian on 2023-11-10.
//

#ifndef EPIC_GACCO_BENCHMARKS_TPCC_GPU_EXECUTOR_H
#define EPIC_GACCO_BENCHMARKS_TPCC_GPU_EXECUTOR_H

#include <gacco/benchmarks/tpcc_executor.h>

namespace gacco::tpcc {

class GpuExecutor : public Executor
{
public:
    GpuExecutor(TpccRecords records, Executor::TpccTableLocks table_locks, TxnArray<TpccTxnParam> txn, epic::tpcc::TpccConfig config)
        : Executor(records, table_locks, txn, config){};
        ~GpuExecutor() override = default;
    void execute(uint32_t epoch) override;
};

}

#endif // EPIC_GACCO_BENCHMARKS_TPCC_GPU_EXECUTOR_H
