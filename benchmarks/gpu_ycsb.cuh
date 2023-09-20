//
// Created by Shujian Qian on 2023-08-19.
//

#ifndef GPU_YCSB_CUH
#define GPU_YCSB_CUH

#include "txn.h"
#include "ycsb.h"
#include "execution_planner.h"

namespace epic {
namespace ycsb {

__host__ __device__ uint32_t gpuCalcNumOps(BaseTxn *txn);
__host__ __device__ uint32_t gpuSubmitOps(BaseTxn *txn, op_t *ops);

} // namespace ycsb
} // namespace epic

#endif // GPU_YCSB_CUH
