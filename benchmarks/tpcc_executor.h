//
// Created by Shujian Qian on 2023-10-25.
//

#ifndef TPCC_EXECUTOR_H
#define TPCC_EXECUTOR_H

#include <benchmarks/tpcc_storage.h>
#include <util_log.h>
#include <txn.h>
#include <benchmarks/tpcc_txn.h>
#include "tpcc_config.h"

namespace epic::tpcc {

template <typename TxnParamArrayType, typename TxnExecPlanArrayType>
class Executor
{
public:
    TpccRecords records;
    TpccVersions versions;
    TxnParamArrayType txn;
    TxnExecPlanArrayType plan;
    TpccConfig config;
    Executor(TpccRecords records, TpccVersions versions, TxnParamArrayType txn, TxnExecPlanArrayType plan,
        TpccConfig config)
        : records(records)
        , versions(versions)
        , txn(txn)
        , plan(plan)
        , config(config)
    {}
    virtual ~Executor() = default;

    virtual void execute(uint32_t epoch)
    {
        auto &logger = Logger::GetInstance();
        logger.Error("Executor::execute() is not implemented.");
    };
};

} // namespace epic::tpcc

#endif // TPCC_EXECUTOR_H
