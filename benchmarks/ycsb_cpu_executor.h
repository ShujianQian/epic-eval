//
// Created by Shujian Qian on 2023-12-05.
//

#ifndef EPIC_BENCHMARKS_YCSB_CPU_EXECUTOR_H
#define EPIC_BENCHMARKS_YCSB_CPU_EXECUTOR_H

#include <benchmarks/ycsb_executor.h>

namespace epic::ycsb {

class CpuExecutor : public Executor
{
public:
    CpuExecutor(YcsbRecordArrType records, YcsbVersionArrType versions, TxnArray<YcsbTxnParam> txn,
        TxnArray<YcsbExecPlan> plan, YcsbConfig config)
        : Executor(records, versions, txn, plan, config){};
    ~CpuExecutor() override = default;

    void execute(uint32_t epoch, uint32_t * pver_sync_expected = nullptr, uint32_t *pver_sync_counter = nullptr) override;
private:
    void executionWorker(uint32_t epoch, uint32_t thread_id);
};

}

#endif // EPIC_BENCHMARKS_YCSB_CPU_EXECUTOR_H
