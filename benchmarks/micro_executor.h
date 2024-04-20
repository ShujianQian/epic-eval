//
// Created by Shujian Qian on 2024-04-20.
//

#ifndef MICRO_EXECUTOR_H
#define MICRO_EXECUTOR_H

#include <gpu_storage.cuh>
#include <util_warp_memory.cuh>
#include <benchmarks/micro_config.h>

namespace epic::micro {

namespace detail {

extern __device__ uint32_t txn_counter; /* used for scheduling txns among threads */
extern const uint32_t zero;

static void __global__ microExecKernel(MicroRecord *records, MicroVersion *versions, GpuTxnArray prev_params, bool *prev_retry, GpuTxnArray curr_params, bool *curr_retry, GpuTxnArray execution_plan, uint32_t num_txns, uint32_t epoch_id)
{
    constexpr uint32_t leader_lane = 0;
    constexpr uint32_t all_lanes_mask = 0xffffffffu;
    constexpr uint32_t num_warps = 512 / 32;

    __shared__ uint8_t txn_param[num_warps][BaseTxnSize<MicroTxnParams>::value];
    __shared__ uint8_t exec_plan[num_warps][BaseTxnSize<MicroTxnExecPlan>::value];
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

    uint32_t warp_txn_id;
    if (lane_id == leader_lane)
    {
        warp_txn_id = atomicAdd(&warp_counter, 1);
    }
    warp_txn_id = __shfl_sync(all_lanes_mask, warp_txn_id, leader_lane);

    if (warp_txn_id >= num_txns * 2)
    {
        return;
    }
    bool is_prev = warp_txn_id < num_txns;
    /* if prev epoch's txn is not retried, then it is not indexed */
    if (is_prev && !prev_retry[warp_txn_id])
    {
        return;
    }

    uint32_t tid_in_epoch = is_prev ? warp_txn_id : warp_txn_id - num_txns;
    BaseTxn *txn_param_ptr = is_prev ? prev_params.getTxn(tid_in_epoch) : curr_params.getTxn(tid_in_epoch);
    BaseTxn *exec_plan_ptr = execution_plan.getTxn(warp_txn_id); // execution plan contains plan for txns from both epochs
    warpMemcpy(reinterpret_cast<uint32_t *>(txn_param[warp_id]), reinterpret_cast<uint32_t *>(txn_param_ptr),
        BaseTxnSize<MicroTxnParams>::value / sizeof(uint32_t), lane_id);
    warpMemcpy(reinterpret_cast<uint32_t *>(exec_plan[warp_id]), reinterpret_cast<uint32_t *>(exec_plan_ptr),
        BaseTxnSize<MicroTxnExecPlan>::value / sizeof(uint32_t), lane_id);

    MicroTxnParams *txn = reinterpret_cast<MicroTxnParams *>(reinterpret_cast<BaseTxn *>(txn_param[warp_id])->data);
    MicroTxnExecPlan *plan = reinterpret_cast<MicroTxnExecPlan *>(reinterpret_cast<BaseTxn *>(exec_plan[warp_id])->data);

    uint32_t data;
    for (int i = 0; i < 10; ++i)
    {
        gpuReadFromTableCoop(records, versions, txn->record_ids[i], plan->read_locs[i], epoch_id, data, lane_id);
        if (!txn->abort)
        {
            // some random data operation
            data ^= warp_txn_id;
        }
        gpuWriteToTableCoop(records, versions, txn->record_ids[i], plan->write_locs[i], epoch_id, data, lane_id);
    }

    if (!is_prev && lane_id == 0)
    {
        curr_retry[tid_in_epoch] = txn->abort;
    }
}

}

class MicroExecutor
{
public:
    MicroConfig config;
    explicit MicroExecutor(MicroConfig config)
        : config(config)
    {}

    void execute(MicroRecord *records, MicroVersion *versions, TxnArray<MicroTxnParams> &prev_params, bool *prev_retry,
        TxnArray<MicroTxnParams> &curr_params, bool *curr_retry, TxnArray<MicroTxnExecPlan> &execution_plan,
        uint32_t epoch_id)
    {
        gpu_err_check(cudaMemcpyToSymbol(detail::txn_counter, &detail::zero, sizeof(uint32_t)));

        auto &logger = Logger::GetInstance();
        constexpr uint32_t block_size = 512;
        constexpr uint32_t warp_size = 32;
        const uint32_t num_blocks = (config.num_txns * 2 * warp_size + block_size - 1) / block_size;

        detail::microExecKernel<<<num_blocks, block_size>>>(records, versions, GpuTxnArray(prev_params), prev_retry,
            GpuTxnArray(curr_params), curr_retry, GpuTxnArray(execution_plan), config.num_txns, epoch_id);

        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaDeviceSynchronize());
    }
};

} // namespace epic::micro

#endif // MICRO_EXECUTOR_H
