//
// Created by Shujian Qian on 2023-08-19.
//

#include "benchmarks/gpu_ycsb.cuh"

namespace epic {
namespace ycsb {

__host__ __device__ uint32_t gpuCalcNumOps(BaseTxn *txn)
{
    uint32_t num_ops = 1;
    return num_ops;
}

__host__ __device__ uint32_t gpuSubmitOps(BaseTxn *txn, op_t *ops)
{
    uint32_t num_ops = 0;
    ops[num_ops++] = 0;
    return num_ops;
}

} // namespace ycsb
} // namespace epic