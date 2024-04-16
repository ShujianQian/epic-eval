//
// Created by Shujian Qian on 2023-12-04.
//

#ifndef EPIC_BENCHMARKS_TPCC_CPU_EXECUTOR_H
#define EPIC_BENCHMARKS_TPCC_CPU_EXECUTOR_H

#include <benchmarks/tpcc_executor.h>

namespace epic::tpcc {

template <typename TxnParamArrayType, typename TxnExecPlanArrayType>
class CpuExecutor : public Executor<TxnParamArrayType, TxnExecPlanArrayType>
{
public:
    using Executor<TxnParamArrayType, TxnExecPlanArrayType>::records;
    using Executor<TxnParamArrayType, TxnExecPlanArrayType>::versions;
    using Executor<TxnParamArrayType, TxnExecPlanArrayType>::config;
    using Executor<TxnParamArrayType, TxnExecPlanArrayType>::txn;
    using Executor<TxnParamArrayType, TxnExecPlanArrayType>::plan;

    CpuExecutor(TpccRecords records, TpccVersions versions, TxnParamArrayType txn, TxnExecPlanArrayType plan,
        TpccConfig config)
        : Executor<TxnParamArrayType, TxnExecPlanArrayType>(records, versions, txn, plan, config){};
    ~CpuExecutor() override = default;

    void execute(uint32_t epoch) override;

private:
    void executionWorker(uint32_t epoch, uint32_t tid);
    void executeTxn(NewOrderTxnParams<FixedSizeTxn> *txn, NewOrderExecPlan<FixedSizeTxn> *plan, uint32_t epoch);
    void executeTxn(PaymentTxnParams *txn, PaymentTxnExecPlan *plan, uint32_t epoch);
    void executeTxn(OrderStatusTxnParams *txn, OrderStatusTxnExecPlan *plan, uint32_t epoch);
    void executeTxn(DeliveryTxnParams *txn, DeliveryTxnExecPlan *plan, uint32_t epoch);
    void executeTxn(StockLevelTxnParams *txn, StockLevelTxnExecPlan *plan, uint32_t epoch);
};

} // namespace epic::tpcc

#endif // EPIC_BENCHMARKS_TPCC_CPU_EXECUTOR_H
