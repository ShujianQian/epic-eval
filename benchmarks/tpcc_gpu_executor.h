//
// Created by Shujian Qian on 2023-10-25.
//

#ifndef TPCC_GPU_EXECUTOR_H
#define TPCC_GPU_EXECUTOR_H

#include <benchmarks/tpcc_executor.h>

namespace epic::tpcc {

template <typename TxnParamArrayType, typename TxnExecPlanArrayType>
class GpuExecutor : public Executor<TxnParamArrayType, TxnExecPlanArrayType>
{
public:
    using Executor<TxnParamArrayType, TxnExecPlanArrayType>::records;
    using Executor<TxnParamArrayType, TxnExecPlanArrayType>::versions;
    using Executor<TxnParamArrayType, TxnExecPlanArrayType>::config;
    using Executor<TxnParamArrayType, TxnExecPlanArrayType>::txn;
    using Executor<TxnParamArrayType, TxnExecPlanArrayType>::plan;

    GpuExecutor(TpccRecords records, TpccVersions versions, TxnParamArrayType txn, TxnExecPlanArrayType plan,
        TpccConfig config)
        : Executor<TxnParamArrayType, TxnExecPlanArrayType>(records, versions, txn, plan, config){};
    ~GpuExecutor() override = default;

    void execute(uint32_t epoch) override;
};

} // namespace epic::tpcc

#endif // TPCC_GPU_EXECUTOR_H
