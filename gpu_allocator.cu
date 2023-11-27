//
// Created by Shujian Qian on 2023-08-09.
//

#include "gpu_allocator.h"

#include "util_gpu_error_check.cuh"
#include "util_log.h"
#include "util_math.h"

namespace epic {
void *GpuAllocator::Allocate(size_t size)
{
    void *ptr;
    gpu_err_check(cudaMalloc(&ptr, size));
    gpu_err_check(cudaMemset(ptr, 0, size));
    return ptr;
}

void GpuAllocator::Free(void *ptr)
{
    gpu_err_check(cudaFree(ptr));
}

void GpuAllocator::PrintMemoryInfo()
{
    size_t free, total;
    gpu_err_check(cudaMemGetInfo(&free, &total));
    auto &logger = Logger::GetInstance();
    logger.Info("GPU memory usage: {} / {}", formatSizeBytes(total - free), formatSizeBytes(total));
}
} // namespace epic