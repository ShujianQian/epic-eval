//
// Created by Shujian Qian on 2023-11-22.
//

#include <benchmarks/ycsb_gpu_executor.h>
#include <gpu_txn.cuh>
#include <util_warp_memory.cuh>
#include "gpu_storage.cuh"

namespace epic::ycsb {

namespace {

constexpr uint32_t block_size = 128;
constexpr uint32_t num_warps = block_size / kDeviceWarpSize;

__device__ uint32_t txn_counter = 0; /* used for scheduling txns among threads */
const uint32_t zero = 0;

__device__ char *op_str[] = {"READ", "FULL_READ", "UPDATE", "READ_MODIFY_WRITE", "FULL_READ_MODIFY_WRITE", "INSERT"};

__device__ char *YcsbOpTypeToString(YcsbOpType type)
{
    return op_str[static_cast<uint8_t>(type)];
}

__global__ void gpuExecKernel(YcsbConfig config, void *records, void *versions, GpuTxnArray txns, GpuTxnArray plans,
    uint32_t num_txns, uint32_t epoch)
{
    constexpr uint32_t leader_lane = 0;
    constexpr uint32_t all_lanes_mask = 0xffffffffu;

    __shared__ uint8_t txn_param[num_warps][BaseTxnSize<YcsbTxnParam>::value];
    __shared__ uint8_t exec_plan[num_warps][BaseTxnSize<YcsbExecPlan>::value];
    static_assert(BaseTxnSize<YcsbTxnParam>::value % sizeof(uint32_t) == 0, "Cannot be copied in 32-bit words");
    static_assert(BaseTxnSize<YcsbExecPlan>::value % sizeof(uint32_t) == 0, "Cannot be copied in 32-bit words");
    __shared__ uint32_t warp_counter;

    constexpr size_t shared_memory_size_per_sm = 64 * 1024;
    constexpr size_t resident_threads_per_sm = 2048;
    constexpr size_t resident_blocks_per_sm = resident_threads_per_sm / block_size;
    constexpr size_t shared_memory_size_per_block = shared_memory_size_per_sm / resident_blocks_per_sm;
    static_assert(sizeof(txn_param) + sizeof(exec_plan) + sizeof(warp_counter) <= shared_memory_size_per_sm,
        "Not enough shared memory");

    uint32_t warp_id = threadIdx.x / kDeviceWarpSize;
    uint32_t lane_id = threadIdx.x % kDeviceWarpSize;
    /* one thread loads txn id for the entire warp */
    if (threadIdx.x == 0)
    {
        warp_counter = atomicAdd(&txn_counter, num_warps);
    }

    __syncthreads();
    /* warp cooperative execution afterward */

    uint32_t warp_txn_id;
    if (lane_id == leader_lane)
    {
        warp_txn_id = atomicAdd(&warp_counter, 1);
    }
    warp_txn_id = __shfl_sync(all_lanes_mask, warp_txn_id, leader_lane);
    if (warp_txn_id >= num_txns)
    {
        return;
    }

    /* load txn param and exec plan into shared memory */
    BaseTxn *txn_param_ptr = txns.getTxn(warp_txn_id);
    warpMemcpy(reinterpret_cast<uint32_t *>(txn_param[warp_id]), reinterpret_cast<uint32_t *>(txn_param_ptr),
        BaseTxnSize<YcsbTxnParam>::value / sizeof(uint32_t), lane_id);
    BaseTxn *exec_plan_ptr = plans.getTxn(warp_txn_id);
    warpMemcpy(reinterpret_cast<uint32_t *>(exec_plan[warp_id]), reinterpret_cast<uint32_t *>(exec_plan_ptr),
        BaseTxnSize<YcsbExecPlan>::value / sizeof(uint32_t), lane_id);
    __syncwarp();

    YcsbTxnParam *txn = reinterpret_cast<YcsbTxnParam *>(reinterpret_cast<BaseTxn *>(txn_param[warp_id])->data);
    YcsbExecPlan *plan = reinterpret_cast<YcsbExecPlan *>(reinterpret_cast<BaseTxn *>(exec_plan[warp_id])->data);

    uint32_t data;
    for (int i = 0; i < 10; ++i)
    {
        uint32_t record_id = txn->record_ids[i] * 10 + txn->field_ids[i];
        switch (txn->ops[i])
        {
        case YcsbOpType::READ:
            //            if (warp_txn_id < 100 && lane_id == leader_lane)
            //            {
            //                printf("txn[%05u][%u] read cr_id[%u] record_id[%u] field_id[%u] loc[%u]\n", warp_txn_id,
            //                i, record_id,
            //                    txn->record_ids[i], txn->field_ids[i], plan->plans[i].read_plan.read_loc);
            //            }
            gpuReadFromTableCoop(reinterpret_cast<YcsbFieldRecords *>(records),
                reinterpret_cast<YcsbFieldVersions *>(versions), record_id, plan->plans[i].read_plan.read_loc, epoch,
                data, lane_id);
            break;
        case YcsbOpType::UPDATE:
            //            if (warp_txn_id < 100 && lane_id == leader_lane)
            //            {
            //                printf("txn[%05u][%u] update cr_id[%u] record_id[%u] field_id[%u] loc[%u]\n", warp_txn_id,
            //                i, record_id,
            //                    txn->record_ids[i], txn->field_ids[i], plan->plans[i].update_plan.write_loc);
            //            }
            gpuWriteToTableCoop(reinterpret_cast<YcsbFieldRecords *>(records),
                reinterpret_cast<YcsbFieldVersions *>(versions), record_id, plan->plans[i].update_plan.write_loc, epoch,
                data, lane_id);
            break;
        case YcsbOpType::READ_MODIFY_WRITE:
            gpuReadFromTableCoop(reinterpret_cast<YcsbFieldRecords *>(records),
                reinterpret_cast<YcsbFieldVersions *>(versions), record_id,
                plan->plans[i].read_modify_write_plan.read_loc, epoch, data, lane_id);
            gpuWriteToTableCoop(reinterpret_cast<YcsbFieldRecords *>(records),
                reinterpret_cast<YcsbFieldVersions *>(versions), record_id,
                plan->plans[i].read_modify_write_plan.write_loc, epoch, data, lane_id);
            break;
        default:
            printf("Invalid op type: %s\n", YcsbOpTypeToString(txn->ops[i]));
            assert(false);
        }
    }
}

__global__ void gpuPiecewiseExecKernel(YcsbConfig config, void *records, void *versions, GpuTxnArray txns,
    GpuTxnArray plans, uint32_t num_txns, uint32_t epoch)
{
    constexpr uint32_t leader_lane = 0;
    constexpr uint32_t all_lanes_mask = 0xffffffffu;

    __shared__ uint32_t warp_counter;

    uint32_t warp_id = threadIdx.x / kDeviceWarpSize;
    uint32_t lane_id = threadIdx.x % kDeviceWarpSize;
    /* one thread loads txn id for the entire warp */
    if (threadIdx.x == 0)
    {
        warp_counter = atomicAdd(&txn_counter, num_warps);
    }

    __syncthreads();
    /* warp cooperative execution afterward */

    uint32_t warp_piece_id;
    if (lane_id == leader_lane)
    {
        warp_piece_id = atomicAdd(&warp_counter, 1);
    }
    warp_piece_id = __shfl_sync(all_lanes_mask, warp_piece_id, leader_lane);
    if (warp_piece_id >= num_txns * 10)
    {
        return;
    }

    uint32_t warp_txn_id = warp_piece_id / 10;
    warp_piece_id = warp_piece_id % 10;

    BaseTxn *txn_param_ptr = txns.getTxn(warp_txn_id);
    BaseTxn *exec_plan_ptr = plans.getTxn(warp_txn_id);
    YcsbTxnParam *txn = reinterpret_cast<YcsbTxnParam *>(reinterpret_cast<BaseTxn *>(txn_param_ptr)->data);
    YcsbExecPlan *plan = reinterpret_cast<YcsbExecPlan *>(reinterpret_cast<BaseTxn *>(exec_plan_ptr)->data);
    YcsbExecPlan::Plan *piece_plan = &plan->plans[warp_piece_id];

    uint32_t data = 0;

    switch (txn->ops[warp_piece_id])
    {
    case YcsbOpType::READ: {
        uint32_t record_id = txn->record_ids[warp_piece_id] * 10 + txn->field_ids[warp_piece_id];
        gpuReadFromTableCoop(reinterpret_cast<YcsbFieldRecords *>(records),
            reinterpret_cast<YcsbFieldVersions *>(versions), record_id, plan->plans[warp_piece_id].read_plan.read_loc,
            epoch, data, lane_id);
        break;
    }
    case YcsbOpType::FULL_READ: {
        uint32_t record_id = txn->record_ids[warp_piece_id] * 10;
        uint32_t field_id = record_id + lane_id;
        gpuReadMultipleFromTableCoop(reinterpret_cast<YcsbFieldRecords *>(records),
            reinterpret_cast<YcsbFieldVersions *>(versions), field_id,
            plan->plans[warp_piece_id].full_read_plan.read_locs[min(lane_id, 9u)], epoch, data, lane_id, 0x3ffu);
        break;
    }
    case YcsbOpType::UPDATE: {
        uint32_t record_id = txn->record_ids[warp_piece_id] * 10 + txn->field_ids[warp_piece_id];
        gpuWriteToTableCoop(reinterpret_cast<YcsbFieldRecords *>(records),
            reinterpret_cast<YcsbFieldVersions *>(versions), record_id,
            plan->plans[warp_piece_id].update_plan.write_loc, epoch, data, lane_id);
        break;
    }
    case YcsbOpType::READ_MODIFY_WRITE: {
        uint32_t record_id = txn->record_ids[warp_piece_id] * 10 + txn->field_ids[warp_piece_id];
        gpuReadFromTableCoop(reinterpret_cast<YcsbFieldRecords *>(records),
            reinterpret_cast<YcsbFieldVersions *>(versions), record_id,
            plan->plans[warp_piece_id].read_modify_write_plan.read_loc, epoch, data, lane_id);
        gpuWriteToTableCoop(reinterpret_cast<YcsbFieldRecords *>(records),
            reinterpret_cast<YcsbFieldVersions *>(versions), record_id,
            plan->plans[warp_piece_id].read_modify_write_plan.write_loc, epoch, data, lane_id);
        break;
    }
    case YcsbOpType::FULL_READ_MODIFY_WRITE: {
        uint32_t record_id = txn->record_ids[warp_piece_id] * 10;
        uint32_t field_id = record_id + lane_id;
        gpuReadMultipleFromTableCoop(reinterpret_cast<YcsbFieldRecords *>(records),
            reinterpret_cast<YcsbFieldVersions *>(versions), field_id,
            plan->plans[warp_piece_id].full_read_modify_write_plan.read_locs[min(lane_id, 9u)], epoch, data, lane_id,
            0x3ffu);
        field_id = record_id + txn->field_ids[warp_piece_id];
        gpuWriteToTableCoop(reinterpret_cast<YcsbFieldRecords *>(records),
            reinterpret_cast<YcsbFieldVersions *>(versions), field_id,
            plan->plans[warp_piece_id].full_read_modify_write_plan.write_loc, epoch, data, lane_id);
        break;
    }
    case YcsbOpType::INSERT: {
        uint32_t record_id = txn->record_ids[warp_piece_id] * 10;
        for (int i = 0; i < 10; ++i)
        {
            uint32_t field_id = record_id + i;
            gpuWriteToTableCoop(reinterpret_cast<YcsbFieldRecords *>(records),
                reinterpret_cast<YcsbFieldVersions *>(versions), field_id,
                plan->plans[warp_piece_id].insert_full_plan.write_locs[i], epoch, data, lane_id);
        }
        break;
    }
    default:
        assert(false);
    }
    if (lane_id == leader_lane)
    {
        txn->record_ids[warp_piece_id] = data; /* to prevent compiler from optimizing out data read */
    }
}

__global__ void gpuNoSplitPiecewiseExecKernel(YcsbConfig config, void *records, void *versions, GpuTxnArray txns,
    GpuTxnArray plans, uint32_t num_txns, uint32_t epoch)
{
    constexpr uint32_t leader_lane = 0;
    constexpr uint32_t all_lanes_mask = 0xffffffffu;

    __shared__ uint32_t warp_counter;

    uint32_t warp_id = threadIdx.x / kDeviceWarpSize;
    uint32_t lane_id = threadIdx.x % kDeviceWarpSize;
    /* one thread loads txn id for the entire warp */
    if (threadIdx.x == 0)
    {
        warp_counter = atomicAdd(&txn_counter, num_warps);
    }

    __syncthreads();
    /* warp cooperative execution afterward */

    uint32_t warp_piece_id;
    if (lane_id == leader_lane)
    {
        warp_piece_id = atomicAdd(&warp_counter, 1);
    }
    warp_piece_id = __shfl_sync(all_lanes_mask, warp_piece_id, leader_lane);
    if (warp_piece_id >= num_txns * 10)
    {
        return;
    }

    uint32_t warp_txn_id = warp_piece_id / 10;
    warp_piece_id = warp_piece_id % 10;

    BaseTxn *txn_param_ptr = txns.getTxn(warp_txn_id);
    BaseTxn *exec_plan_ptr = plans.getTxn(warp_txn_id);
    YcsbTxnParam *txn = reinterpret_cast<YcsbTxnParam *>(reinterpret_cast<BaseTxn *>(txn_param_ptr)->data);
    YcsbExecPlan *plan = reinterpret_cast<YcsbExecPlan *>(reinterpret_cast<BaseTxn *>(exec_plan_ptr)->data);
    YcsbExecPlan::Plan *piece_plan = &plan->plans[warp_piece_id];

    YcsbRecords *records_ptr = reinterpret_cast<YcsbRecords *>(records);
    YcsbVersions *versions_ptr = reinterpret_cast<YcsbVersions *>(versions);

    uint32_t data = 0;

    switch (txn->ops[warp_piece_id])
    {
    case YcsbOpType::READ: {
        uint32_t record_id = txn->record_ids[warp_piece_id];
        gpuReadFromTableCoop(records_ptr, versions_ptr, record_id, plan->plans[warp_piece_id].read_plan.read_loc, epoch,
            data, lane_id, sizeof(YcsbValue::data[0]) / sizeof(uint32_t) * txn->field_ids[warp_piece_id],
            sizeof(YcsbValue::data[0]) / sizeof(uint32_t));
        break;
    }
    case YcsbOpType::FULL_READ: {
        uint32_t record_id = txn->record_ids[warp_piece_id];
        gpuReadFromTableCoop(
            records_ptr, versions_ptr, record_id, plan->plans[warp_piece_id].read_plan.read_loc, epoch, data, lane_id);
        break;
    }
    case YcsbOpType::UPDATE: {
        uint32_t record_id = txn->record_ids[warp_piece_id];
        gpuReadFromTableCoop(records_ptr, versions_ptr, record_id, plan->plans[warp_piece_id].copy_update_plan.read_loc,
            epoch, data, lane_id);
        gpuWriteToTableCoop(records_ptr, versions_ptr, record_id, plan->plans[warp_piece_id].copy_update_plan.write_loc,
            epoch, data, lane_id);
        break;
    }
    case YcsbOpType::READ_MODIFY_WRITE:
    case YcsbOpType::FULL_READ_MODIFY_WRITE: {
        uint32_t record_id = txn->record_ids[warp_piece_id];
        gpuReadFromTableCoop(records_ptr, versions_ptr, record_id,
            plan->plans[warp_piece_id].read_modify_write_plan.read_loc, epoch, data, lane_id);
        gpuWriteToTableCoop(records_ptr, versions_ptr, record_id,
            plan->plans[warp_piece_id].read_modify_write_plan.write_loc, epoch, data, lane_id);
        break;
    }
    case YcsbOpType::INSERT: {
        uint32_t record_id = txn->record_ids[warp_piece_id];
        gpuWriteToTableCoop(records_ptr, versions_ptr, record_id, plan->plans[warp_piece_id].insert_plan.write_loc,
            epoch, data, lane_id);
        break;
    }
    default:
        printf("Invalid op type: %s\n", YcsbOpTypeToString(txn->ops[warp_piece_id]));
        assert(false);
    }

    if (lane_id == leader_lane)
    {
        txn->record_ids[warp_piece_id] = data; /* to prevent compiler from optimizing out data read */
    }
}

} // namespace

void GpuExecutor::execute(uint32_t epoch)
{
    /* clear the txn_counter */
    gpu_err_check(cudaMemcpyToSymbol(txn_counter, &zero, sizeof(uint32_t)));

    uint32_t num_blocks = (config.num_txns * 10 * kDeviceWarpSize + block_size - 1) / block_size;
    if (config.split_field)
    {
        gpuPiecewiseExecKernel<<<num_blocks, block_size>>>(config,
            reinterpret_cast<void *>(std::get<YcsbFieldRecords *>(records)),
            reinterpret_cast<void *>(std::get<YcsbFieldVersions *>(versions)), GpuTxnArray(txn), GpuTxnArray(plan),
            config.num_txns, epoch);
    }
    else
    {
        gpuNoSplitPiecewiseExecKernel<<<num_blocks, block_size>>>(config,
            reinterpret_cast<void *>(std::get<YcsbRecords *>(records)),
            reinterpret_cast<void *>(std::get<YcsbVersions *>(versions)), GpuTxnArray(txn), GpuTxnArray(plan),
            config.num_txns, epoch);
    }

    gpu_err_check(cudaPeekAtLastError());
    gpu_err_check(cudaDeviceSynchronize());
}

} // namespace epic::ycsb