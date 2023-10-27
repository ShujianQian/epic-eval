//
// Created by Shujian Qian on 2023-08-09.
//

#include "gpu_allocator.h"

#include "util_gpu_error_check.cuh"

namespace epic {
void *GpuAllocator::Allocate(size_t size)
{
    void *ptr;
    gpu_err_check(cudaMalloc(&ptr, size));
    return ptr;
}

void GpuAllocator::Free(void *ptr)
{
    gpu_err_check(cudaFree(ptr));
}
} // namespace epic