//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_EXECUTOR_H
#define EPIC_BENCHMARKS_YCSB_EXECUTOR_H

#include <txn.h>
#include <benchmarks/ycsb_txn.h>
#include <benchmarks/ycsb_storage.h>
#include <benchmarks/ycsb_config.h>

namespace epic::ycsb {

class Executor
{
public:
    YcsbRecordArrType records;
    YcsbVersionArrType versions;
    TxnArray<YcsbTxnParam> txn;
    TxnArray<YcsbExecPlan> plan;
    YcsbConfig config;
    Executor(YcsbRecordArrType records, YcsbVersionArrType versions, TxnArray<YcsbTxnParam> txn,
        TxnArray<YcsbExecPlan> plan, YcsbConfig config)
        : records(records)
        , versions(versions)
        , txn(txn)
        , plan(plan)
        , config(config)
    {}
    virtual ~Executor() = default;
    virtual void execute(uint32_t epoch, uint32_t *pver_sync_expected = nullptr, uint32_t *pver_sync_counter = nullptr)
    {
        throw std::runtime_error("epic::ycsb::Executor::execute() is not implemented.");
    };
    virtual void printStat() const
    {
        throw std::runtime_error("epic::ycsb::Executor::printStat() is not implemented.");
    };
};

} // namespace epic::ycsb

#endif // EPIC_BENCHMARKS_YCSB_EXECUTOR_H
