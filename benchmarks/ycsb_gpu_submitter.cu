//
// Created by Shujian Qian on 2023-11-22.
//

#include <thrust/for_each.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <cub/cub.cuh>
#include <cub/device/device_scan.cuh>

#include <benchmarks/ycsb_gpu_submitter.h>
#include <gpu_txn.cuh>
#include <benchmarks/ycsb_config.h>
#include <execution_planner.h>

namespace epic::ycsb {

namespace {

void __global__ prepareSubmitYcsbTxn(YcsbConfig config, GpuTxnArray txns, uint32_t *num_ops, uint32_t num_txns)
{

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns)
    {
        return;
    }
    BaseTxn *base_txn_ptr = txns.getTxn(tid);
    YcsbTxnParam *txn = reinterpret_cast<YcsbTxnParam *>(base_txn_ptr->data);
    uint32_t ops = 0;
    for (auto &op : txn->ops)
    {
        switch (op)
        {
        case YcsbOpType::READ:
            ops += 1;
            break;
        case YcsbOpType::UPDATE:
            ops += config.split_field ? 1 : 2; /* if not split, update is RMW */
            break;
        case YcsbOpType::READ_MODIFY_WRITE:
            ops += 2;
            break;
        case YcsbOpType::FULL_READ:
            ops += config.split_field ? 10 : 1;
            break;
        case YcsbOpType::FULL_READ_MODIFY_WRITE:
            ops += config.split_field ? 11 : 2;
            break;
        case YcsbOpType::INSERT:
            ops += config.split_field ? 10 : 1;
            break;
        }
    }
    num_ops[tid] = ops;
}

void __global__ submitYcsbTxn(YcsbConfig config, GpuTxnArray txns, uint32_t *offset, op_t *ops, uint32_t num_txns)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns)
    {
        return;
    }
    BaseTxn *base_txn_ptr = txns.getTxn(tid);
    YcsbTxnParam *txn = reinterpret_cast<YcsbTxnParam *>(base_txn_ptr->data);
    uint32_t op_idx = offset[tid];

    for (int i = 0; i < 10; ++i)
    {
        switch (txn->ops[i])
        {
        case YcsbOpType::READ:
            ops[op_idx++] =
                CREATE_OP(config.split_field ? txn->record_ids[i] * 10 + txn->field_ids[i] : txn->record_ids[i], tid,
                    read_op, offsetof(YcsbExecPlan, plans[i].read_plan.read_loc) / sizeof(uint32_t));
            break;
        case YcsbOpType::UPDATE:
            if (config.split_field)
            {
                ops[op_idx++] = CREATE_OP(txn->record_ids[i] * 10 + txn->field_ids[i], tid, write_op,
                                          offsetof(YcsbExecPlan, plans[i].update_plan.write_loc) / sizeof(uint32_t));
            }
            else
            {
                ops[op_idx++] = CREATE_OP(txn->record_ids[i], tid, read_op,
                                          offsetof(YcsbExecPlan, plans[i].copy_update_plan.read_loc) / sizeof(uint32_t));
                ops[op_idx++] = CREATE_OP(txn->record_ids[i], tid, write_op,
                                          offsetof(YcsbExecPlan, plans[i].copy_update_plan.write_loc) / sizeof(uint32_t));
            }
            break;
        case YcsbOpType::READ_MODIFY_WRITE:
            ops[op_idx++] =
                CREATE_OP(config.split_field ? txn->record_ids[i] * 10 + txn->field_ids[i] : txn->record_ids[i], tid,
                    read_op, offsetof(YcsbExecPlan, plans[i].read_modify_write_plan.read_loc) / sizeof(uint32_t));
            ops[op_idx++] =
                CREATE_OP(config.split_field ? txn->record_ids[i] * 10 + txn->field_ids[i] : txn->record_ids[i], tid,
                    write_op, offsetof(YcsbExecPlan, plans[i].read_modify_write_plan.write_loc) / sizeof(uint32_t));
            break;
        case YcsbOpType::FULL_READ:
            if (config.split_field)
            {
                for (int j = 0; j < 10; ++j)
                {
                    ops[op_idx++] = CREATE_OP(txn->record_ids[i] * 10 + j, tid, read_op,
                        offsetof(YcsbExecPlan, plans[i].full_read_plan.read_locs[j]) / sizeof(uint32_t));
                }
            }
            else
            {
                ops[op_idx++] = CREATE_OP(txn->record_ids[i], tid, read_op,
                    offsetof(YcsbExecPlan, plans[i].read_plan.read_loc) / sizeof(uint32_t));
            }
            break;
        case YcsbOpType::FULL_READ_MODIFY_WRITE:
            if (config.split_field)
            {
                for (int j = 0; j < 10; ++j)
                {
                    ops[op_idx++] = CREATE_OP(txn->record_ids[i] * 10 + j, tid, read_op,
                        offsetof(YcsbExecPlan, plans[i].full_read_modify_write_plan.read_locs[j]) / sizeof(uint32_t));
                }
                ops[op_idx++] = CREATE_OP(txn->record_ids[i] * 10 + txn->field_ids[i], tid, write_op,
                    offsetof(YcsbExecPlan, plans[i].full_read_modify_write_plan.write_loc) / sizeof(uint32_t));
            }
            else
            {
                ops[op_idx++] = CREATE_OP(txn->record_ids[i], tid, read_op,
                    offsetof(YcsbExecPlan, plans[i].read_modify_write_plan.read_loc) / sizeof(uint32_t));
                ops[op_idx++] = CREATE_OP(txn->record_ids[i], tid, write_op,
                    offsetof(YcsbExecPlan, plans[i].read_modify_write_plan.write_loc) / sizeof(uint32_t));
            }
            break;
        case YcsbOpType::INSERT:
            if (config.split_field)
            {
                for (int j = 0; j < 10; ++j)
                {
                    ops[op_idx++] = CREATE_OP(txn->record_ids[i] * 10 + j, tid, write_op,
                        offsetof(YcsbExecPlan, plans[i].insert_full_plan.write_locs[j]) / sizeof(uint32_t));
                }
            }
            else
            {
                ops[op_idx++] = CREATE_OP(txn->record_ids[i], tid, write_op,
                    offsetof(YcsbExecPlan, plans[i].insert_plan.write_loc) / sizeof(uint32_t));
            }
            break;
        }
    }
}

} // namespace

void YcsbGpuSubmitter::submit(TxnArray<YcsbTxnParam> &txn_array)
{
    auto &logger = Logger::GetInstance();

    constexpr uint32_t block_size = 512;
    uint32_t num_blocks = (config.num_txns + block_size - 1) / block_size;
    prepareSubmitYcsbTxn<<<num_blocks, block_size>>>(
        config, GpuTxnArray(txn_array), submit_dest.d_num_ops, config.num_txns);
    gpu_err_check(cudaGetLastError());

    gpu_err_check(cub::DeviceScan::InclusiveSum(submit_dest.temp_storage, submit_dest.temp_storage_bytes,
        submit_dest.d_num_ops, submit_dest.d_op_offsets + 1, txn_array.num_txns));
    gpu_err_check(cudaMemcpyAsync(&submit_dest.curr_num_ops, submit_dest.d_op_offsets + txn_array.num_txns,
        sizeof(uint32_t), cudaMemcpyDeviceToHost));

    submitYcsbTxn<<<num_blocks, block_size>>>(config, GpuTxnArray(txn_array), submit_dest.d_op_offsets,
        reinterpret_cast<op_t *>(submit_dest.d_submitted_ops), config.num_txns);

    gpu_err_check(cudaGetLastError());
    gpu_err_check(cudaDeviceSynchronize());

    logger.Info("Ycsb num ops: {}", submit_dest.curr_num_ops);
}

} // namespace epic::ycsb
