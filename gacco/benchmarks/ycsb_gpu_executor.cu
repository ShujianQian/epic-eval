//
// Created by Shujian Qian on 2023-11-27.
//

#include <gacco/benchmarks/ycsb_gpu_executor.h>

#include <util_gpu_error_check.cuh>
#include <gpu_txn.cuh>

namespace gacco::ycsb {

using epic::BaseTxn;
using epic::GpuTxnArray;

namespace {

__device__ uint32_t txn_counter = 0; /* used for scheduling txns among threads */
const uint32_t zero = 0;

__device__ __forceinline__ void acquireLock(GaccoTableLock lock, uint32_t record_id, uint32_t txn_id)
{
    /* TODO: pull this out to a separate file */
    bool printed = false;
    uint32_t current_lock_holder = atomicAdd(&lock.lock[record_id], 0);
    while (current_lock_holder != txn_id)
    {
        /* spin */
        if (!printed)
        {
            printed = true;
        }
        current_lock_holder = atomicAdd(&lock.lock[record_id], 0);
    }
}

__device__ __forceinline__ void releaseLock(GaccoTableLock lock, uint32_t record_id, uint32_t txn_id)
{
    /* TODO: pull this out to a separate file */
    uint32_t new_lock_offset = atomicAdd(&lock.offset[record_id], 1) + 1;
    __threadfence();
    uint32_t new_lock_holder = atomicAdd(&lock.access[new_lock_offset], 0);
    atomicExch(&lock.lock[record_id], new_lock_holder);
    __threadfence();
}

__device__ __forceinline__ void readData(void *data, uint32_t size, uint32_t &output)
{
    for (uint32_t i = 0; i < size; i++)
    {
        output += static_cast<uint32_t *>(data)[i];
    }
}
__device__ __forceinline__ void writeData(void *data, uint32_t size, uint32_t input)
{
    for (uint32_t i = 0; i < size; i++)
    {
        static_cast<uint32_t *>(data)[i] = input;
        --input;
    }
}

__global__ void gpuExecKernel(
    YcsbConfig config, YcsbValue *records, GaccoTableLock lock, GpuTxnArray txns, uint32_t num_txns)
{
    __shared__ uint32_t block_counter;

    /* one thread loads txn id for the entire warp */
    uint32_t block_size = blockDim.x;
    uint32_t tid_in_block = threadIdx.x;
    if (threadIdx.x == 0)
    {
        block_counter = atomicAdd(&txn_counter, block_size);
    }
    __syncthreads();

    uint32_t txn_id = block_counter + tid_in_block;
    if (txn_id >= num_txns)
    {
        return;
    }

    BaseTxn *txn_param_ptr = txns.getTxn(txn_id);
    YcsbTxnParam *txn_param = reinterpret_cast<YcsbTxnParam *>(txn_param_ptr->data);

    for (int i = 0; i < 10; ++i)
    {
        acquireLock(lock, txn_param->record_ids[i], txn_id);
        releaseLock(lock, txn_param->record_ids[i], txn_id);
    }
}

__global__ void gpuPiecewiseExecKernel(
    YcsbConfig config, YcsbValue *records, GaccoTableLock lock, GpuTxnArray txns, uint32_t num_txns)
{
    __shared__ uint32_t block_counter;

    /* one thread loads txn id for the entire warp */
    uint32_t block_size = blockDim.x;
    uint32_t tid_in_block = threadIdx.x;
    if (threadIdx.x == 0)
    {
        block_counter = atomicAdd(&txn_counter, block_size);
    }
    __syncthreads();

    uint32_t piece_id = block_counter + tid_in_block;
    uint32_t txn_id = piece_id / 10;
    piece_id = piece_id % 10;
    if (txn_id >= num_txns)
    {
        return;
    }

    BaseTxn *txn_param_ptr = txns.getTxn(txn_id);
    YcsbTxnParam *txn_param = reinterpret_cast<YcsbTxnParam *>(txn_param_ptr->data);
    YcsbValue *record = &records[txn_param->record_ids[piece_id]];

    uint32_t data = 0;

    acquireLock(lock, txn_param->record_ids[piece_id], txn_id);
    if (txn_param->ops[piece_id] == epic::ycsb::YcsbOpType::FULL_READ ||
        txn_param->ops[piece_id] == epic::ycsb::YcsbOpType::FULL_READ_MODIFY_WRITE)
    {
        readData(record->data, sizeof(record->data) / sizeof(uint32_t), data);
    }
    else if (txn_param->ops[piece_id] == YcsbOpType::READ || txn_param->ops[piece_id] == YcsbOpType::READ_MODIFY_WRITE)
    {
        readData(&record->data[txn_param->field_ids[piece_id]], sizeof(record->data[0]) / sizeof(uint32_t), data);
    }
    if (txn_param->ops[piece_id] == YcsbOpType::UPDATE || txn_param->ops[piece_id] == YcsbOpType::READ_MODIFY_WRITE ||
        txn_param->ops[piece_id] == YcsbOpType::FULL_READ_MODIFY_WRITE)
    {
        writeData(&record->data[txn_param->field_ids[piece_id]], sizeof(record->data[0]) / sizeof(uint32_t), data);
    }
    else if (txn_param->ops[piece_id] == YcsbOpType::INSERT)
    {
        writeData(record->data, sizeof(record->data) / sizeof(uint32_t), data);
    }
    __threadfence();
    releaseLock(lock, txn_param->record_ids[piece_id], txn_id);
//    txn_param->record_ids[piece_id] = data; /* to prevent compiler from optimizing out data read */
}

} // namespace

void ycsb::YcsbGpuExecutor::execute(uint32_t epoch)
{
    gpu_err_check(cudaMemcpyToSymbol(txn_counter, &zero, sizeof(uint32_t)));

    constexpr uint32_t block_size = 256;

    //    gpuExecKernel<<<(config.num_txns + block_size - 1) / block_size, block_size>>>(
    //        config, records, lock, GpuTxnArray(txn), config.num_txns);
    gpuPiecewiseExecKernel<<<(config.num_txns * 10 + block_size - 1) / block_size, block_size>>>(
        config, records, lock, GpuTxnArray(txn), config.num_txns);
    gpu_err_check(cudaPeekAtLastError());
    gpu_err_check(cudaDeviceSynchronize());
}
} // namespace gacco::ycsb