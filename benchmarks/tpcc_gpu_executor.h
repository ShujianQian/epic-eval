//
// Created by Shujian Qian on 2023-10-25.
//

#ifndef TPCC_GPU_EXECUTOR_H
#define TPCC_GPU_EXECUTOR_H

#include <benchmarks/tpcc_executor.h>

namespace epic::tpcc {

class GpuExecutor : public Executor
{
public:
    GpuExecutor(TpccRecords records, TpccVersions versions, TxnArray<TpccTxnParam> txn, TxnArray<TpccExecPlan> plan,
        TpccConfig config)
        : Executor(records, versions, txn, plan, config){};
    ~GpuExecutor() override = default;

    void execute(uint32_t epoch) override;
};

} // namespace epic::tpcc

#endif // TPCC_GPU_EXECUTOR_H
