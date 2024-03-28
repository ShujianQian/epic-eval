//
// Created by Shujian Qian on 2024-01-25.
//

#include <benchmarks/ycsb_pver_copyer.h>
#include <pver_copyer.cuh>
#include <util_gpu_error_check.cuh>

namespace epic::ycsb {

void copyYcsbPver(
    YcsbRecords *records, YcsbVersions *versions, op_t *d_ops_to_copy, uint32_t *d_ver_to_copy, uint32_t num_copy)
{
    copyPver(records, versions, d_ops_to_copy, d_ver_to_copy, num_copy);
    gpu_err_check(cudaDeviceSynchronize());
}

} // namespace epic::ycsb