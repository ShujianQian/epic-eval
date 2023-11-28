//
// Created by Shujian Qian on 2023-11-27.
//

#ifndef EPIC_GACCO_BENCHMARKS_YCSB_GPU_EXECUTOR_H
#define EPIC_GACCO_BENCHMARKS_YCSB_GPU_EXECUTOR_H

#include <gacco/benchmarks/ycsb_executor.h>

namespace gacco::ycsb {

class YcsbGpuExecutor : public Executor
{
public:
    YcsbGpuExecutor(YcsbValue *records, GaccoTableLock lock, TxnArray<YcsbTxnParam> txn, YcsbConfig config)
        : Executor(records, lock, txn, config){};
    ~YcsbGpuExecutor() override = default;
    void execute(uint32_t epoch) override;
};

} // namespace gacco::ycsb

#endif // EPIC_GACCO_BENCHMARKS_YCSB_GPU_EXECUTOR_H
