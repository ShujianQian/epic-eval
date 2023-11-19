//
// Created by Shujian Qian on 2023-11-08.
//

#ifndef EPIC_GACCO_EXECUTION_PLANNER_H
#define EPIC_GACCO_EXECUTION_PLANNER_H

#include <cstdint>

namespace gacco {

using op_t = uint64_t;

constexpr op_t record_id_mask = 0xFFFFFFFF00000000ull;
constexpr op_t record_id_shift = 32;
constexpr op_t txn_id_mask = 0x00000000FFFFFFFFull;
constexpr op_t txn_id_shift = 0;

#define GACCO_CREATE_OP(record_id, txn_id) (((op_t)(record_id) << record_id_shift) | ((op_t)(txn_id) << txn_id_shift));
#define GACCO_GET_RECORD_ID(op) (((op_t)(op)&record_id_mask) >> record_id_shift)
#define GACCO_GET_TXN_ID(op) (((op_t)(op)&txn_id_mask) >> txn_id_shift)

struct GaccoTableLock
{
    uint32_t *access = nullptr;
    uint32_t *offset = nullptr;
    uint32_t *lock = nullptr;
};

class TableExecutionPlanner
{
public:
    uint32_t *d_num_ops = nullptr;
    uint32_t *d_op_offsets = nullptr;
    op_t *d_submitted_ops = nullptr;
    void *d_scratch_array = nullptr;
    size_t scratch_array_bytes = 0;
    uint32_t curr_num_ops = 0;
    GaccoTableLock table_lock;

    TableExecutionPlanner() = default;
    virtual ~TableExecutionPlanner() = default;
    virtual void Initialize() = 0;
    virtual void InitializeExecutionPlan() = 0;
    virtual void FinishInitialization() = 0;
};
} // namespace gacco

#endif // EPIC_GACCO_EXECUTION_PLANNER_H
