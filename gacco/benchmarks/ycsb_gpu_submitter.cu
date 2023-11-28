//
// Created by Shujian Qian on 2023-11-27.
//

#include <gacco/benchmarks/ycsb_gpu_submitter.h>

#include <util_gpu_error_check.cuh>
#include <gpu_txn.cuh>
#include <txn.h>

namespace gacco::ycsb {

using epic::BaseTxn;
using epic::GpuTxnArray;

namespace {

__global__ void submitYcsbTxn(GpuTxnArray txns, op_t *submit_dest, YcsbConfig config, uint32_t num_txns)
{
    int txn_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (txn_id >= num_txns)
    {
        return;
    }
    BaseTxn *base_txn_ptr = txns.getTxn(txn_id);
    YcsbTxnParam *txn = reinterpret_cast<YcsbTxnParam *>(base_txn_ptr->data);
    uint32_t op_idx_base = txn_id * 10;
    for (int i = 0; i < 10; ++i)
    {
        submit_dest[op_idx_base + i] = GACCO_CREATE_OP(txn->record_ids[i], txn_id);
    }
}

} // namespace

ycsb::YcsbGpuSubmitter::YcsbGpuSubmitter(TableSubmitDest table_submit_dest, YcsbConfig config)
    : YcsbSubmitter(table_submit_dest, config)
{

    cudaStream_t stream;
    gpu_err_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_stream = stream;
}
YcsbGpuSubmitter::~YcsbGpuSubmitter()
{
    if (cuda_stream.has_value())
    {
        gpu_err_check(cudaStreamDestroy(std::any_cast<cudaStream_t>(cuda_stream)));
    }
}
void YcsbGpuSubmitter::submit(TxnArray<YcsbTxnParam> &txn_array)
{
    constexpr uint32_t block_size = 512;
    uint32_t num_blocks = (config.num_txns + block_size - 1) / block_size;
    submitYcsbTxn<<<num_blocks, block_size, 0, std::any_cast<cudaStream_t>(cuda_stream)>>>(
        GpuTxnArray(txn_array), reinterpret_cast<op_t *>(submit_dest.d_submitted_ops), config, config.num_txns);
    submit_dest.curr_num_ops = config.num_txns * 10;

    gpu_err_check(cudaGetLastError());
    gpu_err_check(cudaStreamSynchronize(std::any_cast<cudaStream_t>(cuda_stream)));
}
} // namespace gacco::ycsb