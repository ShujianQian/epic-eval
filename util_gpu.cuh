//
// Created by Shujian Qian on 2024-02-15.
//

#ifndef EPIC__UTIL_GPU_CUH
#define EPIC__UTIL_GPU_CUH

#include <cstdint>

__device__ __forceinline__ uint32_t get_smid(void)
{
    uint32_t ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__ uint64_t get_clock64(void)
{
    uint64_t retval;
    asm volatile("mov.u64 %0, %clock64;" : "=l"(retval));
    return retval;
}

#endif // EPIC__UTIL_GPU_CUH
