//
// Created by Shujian Qian on 2023-07-19.
//

#ifndef EXECUTION_PLANNER_H
#define EXECUTION_PLANNER_H

#include "txn.h"

namespace epic {

using op_t = uint64_t;
using CalcNumOpsFunc = uint32_t (*)(BaseTxn *txn);
using SubmitOpsFunc = void (*)(BaseTxn *txn, op_t *ops);

class ExecutionPlanner
{
public:
    virtual ~ExecutionPlanner() = default;
    virtual void Initialize() = 0;
    virtual void SubmitOps(CalcNumOpsFunc pre_submit_ops_func, SubmitOpsFunc submit_ops_func) = 0;
    virtual void InitializeExecutionPlan() = 0;
    virtual void ScatterOpLocations() = 0;
};

} // namespace epic

#endif // EXECUTION_PLANNER_H
