//
// Created by Shujian Qian on 2024-04-20.
//

#ifndef MICRO_SUBMITTER_H
#define MICRO_SUBMITTER_H

#include <benchmarks/micro_config.h>
#include <benchmarks/micro_txn.h>

namespace epic::micro {

namespace detail {

static void __global__ prepareSubmitMicroTxn(bool *retry, uint32_t *num_ops, uint32_t num_txns)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns * 2)
    {
        return;
    }
    bool is_prev = tid < num_txns;
    /* if prev epoch's txn is not retried, then it is not indexed */
    if (is_prev && !retry[tid])
    {
        num_ops[tid] = 0;
    } else
    {
        num_ops[tid] = 20;
    }


}

static void __global__ submitMicroTxn(GpuTxnArray prev_txns, bool *retry, GpuTxnArray curr_txns, uint32_t *offset, op_t *ops, uint32_t num_txns)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns * 2)
    {
        return;
    }
    bool is_prev = tid < num_txns;
    /* if prev epoch's txn is not retried, then it is not indexed */
    if (is_prev && !retry[tid])
    {
        return;
    }

    uint32_t tid_in_epoch = is_prev ? tid : tid - num_txns;
    BaseTxn *base_txn_ptr = is_prev ? prev_txns.getTxn(tid_in_epoch) : curr_txns.getTxn(tid_in_epoch);
    MicroTxnParams *txn_ptr = reinterpret_cast<MicroTxnParams *>(base_txn_ptr->data);
    uint32_t op_idx = offset[tid];

    for (int i = 0; i < 10; ++i)
    {
        ops[op_idx++] = CREATE_OP(txn_ptr->record_ids[i], tid, read_op, offsetof(MicroTxnExecPlan, read_locs[i]) / sizeof(uint32_t));
        ops[op_idx++] = CREATE_OP(txn_ptr->record_ids[i], tid, write_op, offsetof(MicroTxnExecPlan, write_locs[i]) / sizeof(uint32_t));
    }
}

} // namespace detail

class MicroSubmitter
{
public:
    MicroConfig config;
    struct TableSubmitDest
    {
        uint32_t *d_num_ops = nullptr;
        uint32_t *d_op_offsets = nullptr;
        void *d_submitted_ops = nullptr;
        void *temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        uint32_t *curr_num_ops = nullptr;
    } submit_dest;

    explicit MicroSubmitter(MicroConfig config)
        : config(config)
    {}

    void setSubmitDest(const TableSubmitDest &submit_dest)
    {
        this->submit_dest = submit_dest;
    }

    void submit(TxnArray<MicroTxnParams> &prev_params, bool *retry, TxnArray<MicroTxnParams> &curr_params)
    {
        auto &logger = Logger::GetInstance();
        constexpr uint32_t block_size = 512;
        const uint32_t num_blocks = (config.num_txns * 2 + block_size - 1) / block_size;

        detail::prepareSubmitMicroTxn<<<num_blocks, block_size>>>(retry, submit_dest.d_num_ops, config.num_txns);
        gpu_err_check(cudaGetLastError());

        gpu_err_check(cub::DeviceScan::InclusiveSum(submit_dest.temp_storage, submit_dest.temp_storage_bytes,
            submit_dest.d_num_ops, submit_dest.d_op_offsets + 1, config.num_txns * 2));
        gpu_err_check(cudaMemcpyAsync(submit_dest.curr_num_ops, submit_dest.d_op_offsets + config.num_txns * 2,
            sizeof(uint32_t), cudaMemcpyDeviceToHost));

        detail::submitMicroTxn<<<num_blocks, block_size>>>(GpuTxnArray(prev_params), retry, GpuTxnArray(curr_params),
            submit_dest.d_op_offsets, static_cast<op_t *>(submit_dest.d_submitted_ops), config.num_txns);

        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaDeviceSynchronize());
        logger.Info("Micro num ops: {}", *submit_dest.curr_num_ops);
        logger.Info("Finished indexing transactions");
    }
};

} // namespace epic::micro

#endif // MICRO_SUBMITTER_H
