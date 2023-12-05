//
// Created by Shujian Qian on 2023-12-04.
//

#ifndef EPIC_BENCHMARKS_TPCC_CPU_EXECUTOR_H
#define EPIC_BENCHMARKS_TPCC_CPU_EXECUTOR_H

#include <benchmarks/tpcc_executor.h>

namespace epic::tpcc {

class CpuExecutor : public Executor
{
public:
    CpuExecutor(TpccRecords records, TpccVersions versions, TxnArray<TpccTxnParam> txn, TxnArray<TpccExecPlan> plan,
        TpccConfig config)
        : Executor(records, versions, txn, plan, config){};
    ~CpuExecutor() override = default;

    void execute(uint32_t epoch) override;

private:
    void executionWorker(uint32_t epoch, uint32_t tid);
    void executeTxn(NewOrderTxnParams<FixedSizeTxn> *txn, NewOrderExecPlan<FixedSizeTxn> *plan, uint32_t epoch);
    void executeTxn(PaymentTxnParams *txn, PaymentTxnExecPlan *plan, uint32_t epoch);
};

} // namespace epic::tpcc

#endif // EPIC_BENCHMARKS_TPCC_CPU_EXECUTOR_H
