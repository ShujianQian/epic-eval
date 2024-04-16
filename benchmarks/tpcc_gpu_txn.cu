//
// Created by Shujian Qian on 2024-04-15.
//

#include <benchmarks/tpcc_gpu_txn.cuh>

#include <cub/cub.cuh>

#include <benchmarks/tpcc_txn.h>
#include <util_gpu_error_check.cuh>

namespace epic::tpcc {

TpccPackedTxnArrayBuilder::TpccPackedTxnArrayBuilder(uint32_t num_txns)
    : num_txns(num_txns)
{
    gpu_err_check(cudaMalloc(&txn_sizes, sizeof(uint32_t) * num_txns));
    cub::DeviceScan::InclusiveSum(
        nullptr, temp_storage_bytes, static_cast<uint32_t *>(nullptr), static_cast<uint32_t *>(nullptr), num_txns);
    gpu_err_check(cudaMalloc(&temp_storage, temp_storage_bytes));
}

static void __global__ calcTxnParamsSizes(GpuPackedTxnArray src, uint32_t *txn_sizes, uint32_t num_txns)
{
    uint32_t txn_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (txn_id >= num_txns)
    {
        return;
    }

    BaseTxn *base_txn_ptr = src.getTxn(txn_id);
    uint32_t txn_type = base_txn_ptr->txn_type;
    constexpr uint32_t tpcc_txn_sizes[6] = {
        0,
        BaseTxnSize<NewOrderTxnParams<FixedSizeTxn>>::value,
        BaseTxnSize<PaymentTxnParams>::value,
        BaseTxnSize<OrderStatusTxnParams>::value,
        BaseTxnSize<DeliveryTxnParams>::value,
        BaseTxnSize<StockLevelTxnParams>::value
    };
    txn_sizes[txn_id] = tpcc_txn_sizes[txn_type];
}

template <>
void TpccPackedTxnArrayBuilder::buildPackedTxnArrayGpu(PackedTxnArray<TpccTxn> &src, PackedTxnArray<TpccTxnParam> &dest)
{
    constexpr uint32_t block_size = 512;
    uint32_t num_blocks = (num_txns + block_size - 1) / block_size;
    calcTxnParamsSizes<<<num_blocks, block_size>>>(GpuPackedTxnArray(src), txn_sizes, src.num_txns);
    cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes, txn_sizes, dest.index + 1, num_txns);
    gpu_err_check(cudaDeviceSynchronize());
}

static void __global__ calcTxnExecPlanSizes(GpuPackedTxnArray src, uint32_t *txn_sizes, uint32_t num_txns)
{
    uint32_t txn_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (txn_id >= num_txns)
    {
        return;
    }

    BaseTxn *base_txn_ptr = src.getTxn(txn_id);
    uint32_t txn_type = base_txn_ptr->txn_type;
    constexpr uint32_t tpcc_txn_sizes[6] = {
        0,
        BaseTxnSize<NewOrderExecPlan<FixedSizeTxn>>::value,
        BaseTxnSize<PaymentTxnExecPlan>::value,
        BaseTxnSize<OrderStatusTxnExecPlan>::value,
        BaseTxnSize<DeliveryTxnExecPlan>::value,
        BaseTxnSize<StockLevelTxnExecPlan>::value
    };
    txn_sizes[txn_id] = tpcc_txn_sizes[txn_type];
}

template <>
void TpccPackedTxnArrayBuilder::buildPackedTxnArrayGpu(PackedTxnArray<TpccTxn> &src, PackedTxnArray<TpccExecPlan> &dest)
{
    constexpr uint32_t block_size = 512;
    uint32_t num_blocks = (num_txns + block_size - 1) / block_size;
    calcTxnExecPlanSizes<<<num_blocks, block_size>>>(GpuPackedTxnArray(src), txn_sizes, src.num_txns);
    cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes, txn_sizes, dest.index + 1, num_txns);
    gpu_err_check(cudaDeviceSynchronize());
}

}
