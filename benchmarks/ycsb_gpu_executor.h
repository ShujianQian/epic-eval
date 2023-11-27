//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC__YCSB_GPU_EXECUTOR_H
#define EPIC__YCSB_GPU_EXECUTOR_H

#include "ycsb_executor.h"

namespace epic::ycsb {

class GpuExecutor : public Executor
{
public:
    GpuExecutor(YcsbRecordArrType records, YcsbVersionArrType versions, TxnArray<YcsbTxnParam> txn,
        TxnArray<YcsbExecPlan> plan, YcsbConfig config)
        : Executor(records, versions, txn, plan, config){};
    ~GpuExecutor() override = default;

    void execute(uint32_t epoch) override;
};

}

#endif // EPIC__YCSB_GPU_EXECUTOR_H
