//
// Created by Shujian Qian on 2023-11-22.
//

#include <benchmarks/ycsb_gpu_executor.h>
#include <gpu_txn.cuh>
#include <util_warp_memory.cuh>
#include "gpu_storage.cuh"
#include <util_gpu.cuh>
#include <numeric>
#include <algorithm>

#define EPIC_SINGLE_THREAD_EXEC
#undef EPIC_SINGLE_THREAD_EXEC

namespace epic::ycsb {

namespace {

#ifndef EPIC_SINGLE_THREAD_EXEC
constexpr uint32_t block_size = 128;
#else
constexpr uint32_t block_size = 1024;
#endif
constexpr uint32_t num_warps = block_size / kDeviceWarpSize;

__device__ uint32_t txn_counter = 0; /* used for scheduling txns among threads */
const uint32_t zero = 0;

__device__ uint64_t read_sync_times[1000'000];
__device__ uint64_t write_sync_times[1000'000];
__device__ uint64_t start_times[1000'000];
__device__ uint64_t finish_times[1000'000];
__device__ uint64_t perSMClocks[128];
__device__ uint64_t perSMClocks2[128];

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
    GpuTxnArray plans, uint32_t num_txns, uint32_t epoch, bool sync_pver = false,
    uint32_t *pver_sync_expected = nullptr, uint32_t *pver_sync_counter = nullptr)
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
    uint64_t read_sync_time = 0;
    uint64_t write_sync_time = 0;
    uint64_t start_time, end_time;

    start_time = get_clock64();

    switch (txn->ops[warp_piece_id])
    {
    case YcsbOpType::READ: {
        uint32_t record_id = txn->record_ids[warp_piece_id];
        gpuReadFromTableCoop(records_ptr, versions_ptr, record_id, plan->plans[warp_piece_id].read_plan.read_loc, epoch,
            data, lane_id, sizeof(YcsbValue::data[0]) / sizeof(uint32_t) * txn->field_ids[warp_piece_id],
            sizeof(YcsbValue::data[0]) / sizeof(uint32_t));
        if (sync_pver)
        {
            uint64_t start, end;
            asm volatile("mov.u64 %0, %clock64;" : "=l"(start));
            gpuSyncRecordARead(record_id, plan->plans[warp_piece_id].read_plan.read_loc, pver_sync_counter, lane_id);
            asm volatile("mov.u64 %0, %clock64;" : "=l"(end));
            read_sync_time += end - start;
        }
        break;
    }
    case YcsbOpType::FULL_READ: {
        uint32_t record_id = txn->record_ids[warp_piece_id];
        gpuReadFromTableCoop(
            records_ptr, versions_ptr, record_id, plan->plans[warp_piece_id].read_plan.read_loc, epoch, data, lane_id);
        if (sync_pver)
        {
            uint64_t start, end;
            asm volatile("mov.u64 %0, %clock64;" : "=l"(start));
            gpuSyncRecordARead(record_id, plan->plans[warp_piece_id].read_plan.read_loc, pver_sync_counter, lane_id);
            asm volatile("mov.u64 %0, %clock64;" : "=l"(end));
            read_sync_time += end - start;
        }
        break;
    }
    case YcsbOpType::UPDATE: {
        uint32_t record_id = txn->record_ids[warp_piece_id];
        gpuReadFromTableCoop(records_ptr, versions_ptr, record_id, plan->plans[warp_piece_id].copy_update_plan.read_loc,
            epoch, data, lane_id);
        if (sync_pver)
        {
            uint64_t start, end;
            asm volatile("mov.u64 %0, %clock64;" : "=l"(start));
            gpuSyncRecordARead(
                record_id, plan->plans[warp_piece_id].copy_update_plan.read_loc, pver_sync_counter, lane_id);
            asm volatile("mov.u64 %0, %clock64;" : "=l"(end));
            read_sync_time += end - start;

            asm volatile("mov.u64 %0, %clock64;" : "=l"(start));
            gpuSyncRecordBRW(record_id, plan->plans[warp_piece_id].copy_update_plan.write_loc, pver_sync_expected,
                pver_sync_counter, lane_id);
            asm volatile("mov.u64 %0, %clock64;" : "=l"(end));
            write_sync_time += end - start;
        }
        gpuWriteToTableCoop(records_ptr, versions_ptr, record_id, plan->plans[warp_piece_id].copy_update_plan.write_loc,
            epoch, data, lane_id);
        break;
    }
    case YcsbOpType::READ_MODIFY_WRITE:
    case YcsbOpType::FULL_READ_MODIFY_WRITE: {
        uint32_t record_id = txn->record_ids[warp_piece_id];
        gpuReadFromTableCoop(records_ptr, versions_ptr, record_id,
            plan->plans[warp_piece_id].read_modify_write_plan.read_loc, epoch, data, lane_id);
        if (sync_pver)
        {
            uint64_t start, end;
            asm volatile("mov.u64 %0, %clock64;" : "=l"(start));
            gpuSyncRecordARead(
                record_id, plan->plans[warp_piece_id].read_modify_write_plan.read_loc, pver_sync_counter, lane_id);
            asm volatile("mov.u64 %0, %clock64;" : "=l"(end));
            read_sync_time += end - start;

            asm volatile("mov.u64 %0, %clock64;" : "=l"(start));
            gpuSyncRecordBRW(record_id, plan->plans[warp_piece_id].read_modify_write_plan.write_loc, pver_sync_expected,
                pver_sync_counter, lane_id);
            asm volatile("mov.u64 %0, %clock64;" : "=l"(end));
            write_sync_time += end - start;
        }
        gpuWriteToTableCoop(records_ptr, versions_ptr, record_id,
            plan->plans[warp_piece_id].read_modify_write_plan.write_loc, epoch, data, lane_id);
        break;
    }
    case YcsbOpType::INSERT: {
        uint32_t record_id = txn->record_ids[warp_piece_id];
        if (sync_pver)
        {
            uint64_t start, end;
            asm volatile("mov.u64 %0, %clock64;" : "=l"(start));
            gpuSyncRecordBRW(record_id, plan->plans[warp_piece_id].insert_plan.write_loc, pver_sync_expected,
                pver_sync_counter, lane_id);
            asm volatile("mov.u64 %0, %clock64;" : "=l"(end));
            write_sync_time += end - start;
        }
        gpuWriteToTableCoop(records_ptr, versions_ptr, record_id, plan->plans[warp_piece_id].insert_plan.write_loc,
            epoch, data, lane_id);
        break;
    }
    default:
        printf("Invalid op type: %s\n", YcsbOpTypeToString(txn->ops[warp_piece_id]));
        assert(false);
    }

    end_time = get_clock64();

    if (lane_id == leader_lane)
    {
        txn->record_ids[warp_piece_id] = data; /* to prevent compiler from optimizing out data read */
        uint32_t stat_idx = warp_txn_id * 10 + warp_piece_id;
        write_sync_times[stat_idx] = write_sync_time;
        read_sync_times[stat_idx] = read_sync_time;

        uint32_t smid = get_smid();
        start_times[stat_idx] = start_time - perSMClocks[smid];
        finish_times[stat_idx] = end_time - perSMClocks2[smid];

        //        __threadfence();
        //        if (read_sync_time || write_sync_time) {
        //                printf("txn[%05u][%u] read_sync_time[%lu] write_sync_time[%lu]\n", warp_txn_id, warp_piece_id,
        //                read_sync_times[warp_piece_id], write_sync_times[warp_piece_id]);
        //        }
    }
}

__global__ void gpuNoSplitThreadPiecewiseExecKernel(YcsbConfig config, void *records, void *versions, GpuTxnArray txns,
    GpuTxnArray plans, uint32_t num_txns, uint32_t epoch)
{
    __shared__ uint32_t block_counter;

    uint32_t thread_id = threadIdx.x;
    uint32_t lane_id = threadIdx.x % kDeviceWarpSize;
    /* one thread loads txn id for the entire warp */
    if (threadIdx.x == 0)
    {
        block_counter = atomicAdd(&txn_counter, block_size);
    }

    __syncthreads();
    /* warp cooperative execution afterward */

    uint32_t thread_piece_id;
    thread_piece_id = atomicAdd(&block_counter, 1);
    if (thread_piece_id >= num_txns * 10)
    {
        return;
    }

    uint32_t thread_txn_id = thread_piece_id / 10;
    thread_piece_id = thread_piece_id % 10;

    BaseTxn *txn_param_ptr = txns.getTxn(thread_txn_id);
    BaseTxn *exec_plan_ptr = plans.getTxn(thread_txn_id);
    YcsbTxnParam *txn = reinterpret_cast<YcsbTxnParam *>(reinterpret_cast<BaseTxn *>(txn_param_ptr)->data);
    YcsbExecPlan *plan = reinterpret_cast<YcsbExecPlan *>(reinterpret_cast<BaseTxn *>(exec_plan_ptr)->data);
    YcsbExecPlan::Plan *piece_plan = &plan->plans[thread_piece_id];

    YcsbRecords *records_ptr = reinterpret_cast<YcsbRecords *>(records);
    YcsbVersions *versions_ptr = reinterpret_cast<YcsbVersions *>(versions);

    uint32_t data = 0;

    switch (txn->ops[thread_piece_id])
    {
    case YcsbOpType::READ: {
        uint32_t record_id = txn->record_ids[thread_piece_id];
        gpuReadFromTableThread(records_ptr, versions_ptr, record_id, plan->plans[thread_piece_id].read_plan.read_loc,
            epoch, data, sizeof(YcsbValue::data[0]) / sizeof(uint32_t) * txn->field_ids[thread_piece_id],
            sizeof(YcsbValue::data[0]) / sizeof(uint32_t));
        break;
    }
    case YcsbOpType::FULL_READ: {
        uint32_t record_id = txn->record_ids[thread_piece_id];
        gpuReadFromTableThread(
            records_ptr, versions_ptr, record_id, plan->plans[thread_piece_id].read_plan.read_loc, epoch, data);
        break;
    }
    case YcsbOpType::UPDATE: {
        uint32_t record_id = txn->record_ids[thread_piece_id];
        gpuReadFromTableThread(
            records_ptr, versions_ptr, record_id, plan->plans[thread_piece_id].copy_update_plan.read_loc, epoch, data);
        gpuWriteToTableThread(
            records_ptr, versions_ptr, record_id, plan->plans[thread_piece_id].copy_update_plan.write_loc, epoch, data);
        break;
    }
    case YcsbOpType::READ_MODIFY_WRITE:
    case YcsbOpType::FULL_READ_MODIFY_WRITE: {
        uint32_t record_id = txn->record_ids[thread_piece_id];
        gpuReadFromTableThread(records_ptr, versions_ptr, record_id,
            plan->plans[thread_piece_id].read_modify_write_plan.read_loc, epoch, data);
        gpuWriteToTableThread(records_ptr, versions_ptr, record_id,
            plan->plans[thread_piece_id].read_modify_write_plan.write_loc, epoch, data);
        break;
    }
    case YcsbOpType::INSERT: {
        uint32_t record_id = txn->record_ids[thread_piece_id];
        gpuWriteToTableThread(
            records_ptr, versions_ptr, record_id, plan->plans[thread_piece_id].insert_plan.write_loc, epoch, data);
        break;
    }
    default:
        printf("Invalid op type: %s\n", YcsbOpTypeToString(txn->ops[thread_piece_id]));
        assert(false);
    }
    txn->record_ids[thread_piece_id] = data; /* to prevent compiler from optimizing out data read */
}

__global__ void recordPerSMClock()
{
    /* only one thread per block records the clock */
    if (threadIdx.x != 0)
    {
        return;
    }

    uint32_t smid = get_smid();
    unsigned long long *perSMClock = reinterpret_cast<unsigned long long *>(&perSMClocks[smid]);

    unsigned long long ts = get_clock64();
    atomicMax(perSMClock, ts);
}

__global__ void recordPerSMClock2()
{
    /* only one thread per block records the clock */
    if (threadIdx.x != 0)
    {
        return;
    }

    uint32_t smid = get_smid();
    unsigned long long *perSMClock = reinterpret_cast<unsigned long long *>(&perSMClocks2[smid]);

    unsigned long long ts = get_clock64();
    atomicMax(perSMClock, ts);
}
} // namespace

void GpuExecutor::execute(uint32_t epoch, uint32_t *pver_sync_expected, uint32_t *pver_sync_counter)
{
    /* clear the txn_counter */
    gpu_err_check(cudaMemcpyToSymbol(txn_counter, &zero, sizeof(uint32_t)));

#ifndef EPIC_SINGLE_THREAD_EXEC
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
        recordPerSMClock2<<<640, 128>>>();
        recordPerSMClock<<<640, 128>>>();
        gpuNoSplitPiecewiseExecKernel<<<num_blocks, block_size>>>(config,
            reinterpret_cast<void *>(std::get<YcsbRecords *>(records)),
            reinterpret_cast<void *>(std::get<YcsbVersions *>(versions)), GpuTxnArray(txn), GpuTxnArray(plan),
            config.num_txns, epoch, false, pver_sync_expected, pver_sync_counter);
    }
#else
    uint32_t num_blocks = (config.num_txns * 10 + block_size - 1) / block_size;
    gpuNoSplitThreadPiecewiseExecKernel<<<num_blocks, block_size>>>(config,
        reinterpret_cast<void *>(std::get<YcsbRecords *>(records)),
        reinterpret_cast<void *>(std::get<YcsbVersions *>(versions)), GpuTxnArray(txn), GpuTxnArray(plan),
        config.num_txns, epoch);
#endif

    gpu_err_check(cudaPeekAtLastError());
    gpu_err_check(cudaDeviceSynchronize());

#if 0 // SYNC_STAT
    {
        uint32_t *h_read_sync_expected = new uint32_t[2500000];
        gpu_err_check(
            cudaMemcpy(h_read_sync_expected, pver_sync_expected, sizeof(uint32_t) * 2500000, cudaMemcpyDeviceToHost));

        uint32_t read_sync_expected_hist[64]{};
        for (int i = 0; i < config.num_txns * 10; ++i)
        {
            uint32_t mult = 1;
            uint32_t index = 0;
            while (h_read_sync_expected[i] >= mult && index < 20)
            {
                mult <<= 1;
                ++index;
            }
            ++read_sync_expected_hist[index];
        }

        Logger::GetInstance().Info("Read sync expected histogram:");
        for (int i = 0; i < 20; ++i)
        {
            Logger::GetInstance().Info("{}-{}: {}", (1 << i) >> 1, 1 << i, read_sync_expected_hist[i]);
        }

        delete[] h_read_sync_expected;
    }
#endif
}

void GpuExecutor::printStat() const
{
    //    uint64_t *h_read_sync_times = new uint64_t[config.num_txns * 10];
    //    uint64_t *h_write_sync_times = new uint64_t[config.num_txns * 10];
    uint64_t *h_read_sync_times = nullptr, *h_write_sync_times = nullptr, *h_start_times = nullptr,
             *h_finish_times = nullptr;
    gpu_err_check(cudaMallocHost(&h_read_sync_times, sizeof(uint64_t) * config.num_txns * 10));
    gpu_err_check(cudaMallocHost(&h_write_sync_times, sizeof(uint64_t) * config.num_txns * 10));
    gpu_err_check(cudaMallocHost(&h_start_times, sizeof(uint64_t) * config.num_txns * 10));
    gpu_err_check(cudaMallocHost(&h_finish_times, sizeof(uint64_t) * config.num_txns * 10));

    auto &logger = Logger::GetInstance();

    gpu_err_check(cudaMemcpyFromSymbol(h_read_sync_times, read_sync_times, sizeof(uint64_t) * config.num_txns * 10));
    gpu_err_check(cudaMemcpyFromSymbol(h_write_sync_times, write_sync_times, sizeof(uint64_t) * config.num_txns * 10));
    gpu_err_check(cudaMemcpyFromSymbol(h_start_times, start_times, sizeof(uint64_t) * config.num_txns * 10));
    gpu_err_check(cudaMemcpyFromSymbol(h_finish_times, finish_times, sizeof(uint64_t) * config.num_txns * 10));
    gpu_err_check(cudaDeviceSynchronize());

    uint64_t min_start_time = std::reduce(
        h_start_times, h_start_times + config.num_txns * 10, UINT64_MAX, [](auto a, auto b) { return std::min(a, b); });
    std::transform(h_start_times, h_start_times + config.num_txns * 10, h_start_times,
        [min_start_time](auto a) { return a - min_start_time; });
    std::transform(h_finish_times, h_finish_times + config.num_txns * 10, h_finish_times,
        [min_start_time](auto a) { return a - min_start_time; });

    /* print histogram of read sync times */
    uint32_t read_sync_hist[64]{}, write_sync_hist[64]{};
    uint32_t start_time_hist[64]{}, finish_time_hist[64]{};
    uint32_t execution_time_hist[64]{};
    uint32_t execution_time_linear_hist[256]{};

    for (int i = 0; i < config.num_txns * 10; ++i)
    {
        uint64_t mult = 1;
        uint32_t index = 0;
        while (h_write_sync_times[i] > mult && index < 20)
        {
            mult <<= 1;
            ++index;
        }
        ++write_sync_hist[index];

        mult = 1;
        index = 0;
        while (h_read_sync_times[i] > mult && index < 20)
        {
            mult <<= 1;
            ++index;
        }
        ++read_sync_hist[index];

        mult = 1;
        index = 0;
        while (h_start_times[i] > mult && index < 29)
        {
            mult <<= 1;
            ++index;
        }
        ++start_time_hist[index];

        mult = 1;
        index = 0;
        while (h_finish_times[i] > mult && index < 29)
        {
            mult <<= 1;
            ++index;
        }
        ++finish_time_hist[index];

        mult = 1;
        index = 0;
        while (h_finish_times[i] - h_start_times[i] > mult && index < 29)
        {
            mult <<= 1;
            ++index;
        }
        ++execution_time_hist[index];

        index = 0;
        while (h_finish_times[i] - h_start_times[i] > (index + 1) * 1000 && index < 255)
        {
            ++index;
        }
        ++execution_time_linear_hist[index];
    }

    logger.Info("Read sync times histogram:");
    for (int i = 0; i < 20; ++i)
    {
        logger.Info("{}-{}: {}", 1 << i >> 1, 1 << (i + 1) >> 1, read_sync_hist[i]);
    }

    logger.Info("Write sync times histogram:");
    for (int i = 0; i < 20; ++i)
    {
        logger.Info("{}-{}: {}", 1 << i >> 1, 1 << (i + 1) >> 1, write_sync_hist[i]);
    }

    logger.Info("Start times histogram:");
    for (int i = 0; i < 30; ++i)
    {
        logger.Info("{}-{}: {}", 1 << i >> 1, 1 << (i + 1) >> 1, start_time_hist[i]);
    }

    logger.Info("Finish times histogram:");
    for (int i = 0; i < 30; ++i)
    {
        logger.Info("{}-{}: {}", 1 << i >> 1, 1 << (i + 1) >> 1, finish_time_hist[i]);
    }

    logger.Info("Execution times histogram:");
    for (int i = 0; i < 30; ++i)
    {
        logger.Info("{}-{}: {}", 1 << i >> 1, 1 << (i + 1) >> 1, execution_time_hist[i]);
    }

    logger.Info("Execution times linear histogram:");
    for (int i = 0; i < 256; ++i)
    {
        logger.Info("{}-{}: {}", i * 1000, (i + 1) * 1000, execution_time_linear_hist[i]);
    }

    /* print per SM clock */
    {

        uint64_t *h_perSMClocks = new uint64_t[128];
        uint64_t *h_perSMClocks2 = new uint64_t[128];
        gpu_err_check(cudaMemcpyFromSymbol(h_perSMClocks, perSMClocks, sizeof(uint64_t) * 128));
        gpu_err_check(cudaMemcpyFromSymbol(h_perSMClocks2, perSMClocks2, sizeof(uint64_t) * 128));
        for (int i = 0; i < 90; ++i)
        {
            logger.Info(
                "SM[{}]: {} {} {}", i, h_perSMClocks2[i], h_perSMClocks[i], h_perSMClocks[i] - h_perSMClocks2[i]);
        }
    }

    int device;
    gpu_err_check(cudaGetDevice(&device));

    struct cudaDeviceProp prop;
    gpu_err_check(cudaGetDeviceProperties(&prop, device));
    logger.Info("GPU frequency: {} MHz", prop.clockRate / 1000.0);
    logger.Info("SM count: {}", prop.multiProcessorCount);

    //    delete[] h_read_sync_times;
    //    delete[] h_write_sync_times;
    gpu_err_check(cudaFreeHost(h_read_sync_times));
    gpu_err_check(cudaFreeHost(h_write_sync_times));
    gpu_err_check(cudaFreeHost(h_start_times));
    gpu_err_check(cudaFreeHost(h_finish_times));
}

} // namespace epic::ycsb