//
// Created by Shujian Qian on 2024-01-25.
//

#ifndef EPIC__PVER_COPYER_CUH
#define EPIC__PVER_COPYER_CUH

#include <storage.h>
#include <execution_planner.h>
#include <util_arch.h>
#include <util_gpu_error_check.cuh>

#include <stdio.h>

namespace epic {

namespace {

template<typename ValueType>
__global__ void copyPverKernel(Record<ValueType> *records, Version<ValueType> *versions, op_t *d_ops_to_copy,
    uint32_t *d_ver_to_copy, uint32_t num_copy)
{
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t lane_id = tid % kDeviceWarpSize;
    uint32_t copy_id = tid / kDeviceWarpSize;
    if (copy_id >= num_copy)
    {
        return;
    }

    uint32_t ver_to_copy = GET_RECORD_ID(d_ops_to_copy[copy_id]);
//    if (lane_id == 0) {
//        printf("copying version %u\n", ver_to_copy);
//    }

    uint32_t *from = reinterpret_cast<uint32_t *>(&records[ver_to_copy].value2);
    uint32_t *to = reinterpret_cast<uint32_t *>(&records[ver_to_copy].value1);
    const uint32_t copy_step = sizeof(uint32_t) * kDeviceWarpSize;
    const uint32_t lane_offset = lane_id * sizeof(uint32_t);
    for (uint32_t i = 0; i < sizeof(ValueType); i += copy_step)
    {
        if (lane_offset + i < sizeof(ValueType))
        {
            to[lane_id + i] = from[lane_id + i];
        }
    }
}

} // namespace

template<typename ValueType>
void copyPver(Record<ValueType> *records, Version<ValueType> *versions, op_t *d_ops_to_copy, uint32_t *d_ver_to_copy,
    uint32_t num_copy)
{
#if 0 // DEBUG
    {
        auto &logger = Logger::GetInstance();
        uint32_t debug_num_copy_pver = std::min(num_copy, 100u);
        op_t ops_to_copy[debug_num_copy_pver];
        gpu_err_check(cudaMemcpy(ops_to_copy, d_ops_to_copy, sizeof(op_t) * debug_num_copy_pver,
                                 cudaMemcpyDeviceToHost));
        for (int i = 0; i < debug_num_copy_pver; i++)
        {
            logger.Info("YcsbCopyer op{}: record[{}] txn[{}] rw[{}] offset[{}]",
                        i, GET_RECORD_ID(ops_to_copy[i]), GET_TXN_ID(ops_to_copy[i]), GET_R_W(ops_to_copy[i]),
                        GET_OFFSET(ops_to_copy[i]));
        }
    }
#endif
    const uint32_t block_size = 512;
    const uint32_t copies_per_block = block_size / kDeviceWarpSize;
    const uint32_t grid_size = (num_copy + copies_per_block - 1) / copies_per_block;
    copyPverKernel<<<grid_size, block_size>>>(records, versions, d_ops_to_copy, d_ver_to_copy, num_copy);
}

} // namespace epic

#endif // EPIC__PVER_COPYER_CUH
