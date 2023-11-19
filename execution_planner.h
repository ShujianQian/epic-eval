//
// Created by Shujian Qian on 2023-07-19.
//

#ifndef EXECUTION_PLANNER_H
#define EXECUTION_PLANNER_H

#include <cstdint>
#include <limits>

#include "txn.h"

namespace epic {

enum class OperationT : uint8_t
{
    VERSION_READ,
    VERSION_WRITE,
    RECORD_A_READ,
    RECORD_B_READ,
    RECORD_A_WRITE,
    RECORD_B_WRITE
};

using op_t = uint64_t;
using CalcNumOpsFunc = uint32_t (*)(BaseTxn *txn);
using SubmitOpsFunc = void (*)(BaseTxn *txn, op_t *ops);

constexpr op_t record_id_mask = 0xFFFFFFFF00000000ull;
constexpr op_t record_id_shift = 32;
constexpr op_t txn_id_mask = 0x00000000FFFFF000ull;
constexpr op_t txn_id_shift = 12;
constexpr op_t r_w_mask = 0x0000000000000F00ull;
constexpr op_t r_w_shift = 8;
constexpr op_t offset_mask = 0x00000000000000FFull;
constexpr op_t offset_shift = 0;

constexpr op_t read_op = 0x0ull;
constexpr op_t write_op = 0x1ull;

#define CREATE_OP(record_id, txn_id, r_w, offset) \
    (((op_t)(record_id) << record_id_shift) | ((op_t)(txn_id) << txn_id_shift) | ((op_t)(r_w) << r_w_shift) | ((op_t)(offset) << offset_shift));
#define GET_RECORD_ID(op) (((op_t)(op)&record_id_mask) >> record_id_shift)
#define GET_TXN_ID(op) (((op_t)(op)&txn_id_mask) >> txn_id_shift)
#define GET_R_W(op) (((op_t)(op)&r_w_mask) >> r_w_shift)
#define GET_OFFSET(op) (((op_t)(op)&offset_mask) >> offset_shift)

constexpr uint32_t loc_record_a = std::numeric_limits<uint32_t>::max();
constexpr uint32_t loc_record_b = std::numeric_limits<uint32_t>::max() - 1;

class TableExecutionPlanner
{
public:
    uint32_t *d_num_ops = nullptr;
    uint32_t *d_op_offsets = nullptr;
    void *d_submitted_ops = nullptr;
    void *d_scratch_array = nullptr;
    size_t scratch_array_bytes = 0;
    uint32_t curr_num_ops = 0;


    virtual ~TableExecutionPlanner() = default;
    virtual void Initialize() = 0;
    virtual void SubmitOps(CalcNumOpsFunc pre_submit_ops_func, SubmitOpsFunc submit_ops_func) = 0;
    virtual void InitializeExecutionPlan() = 0;
    virtual void FinishInitialization() = 0;
    virtual void ScatterOpLocations() = 0;
};

} // namespace epic

#endif // EXECUTION_PLANNER_H
