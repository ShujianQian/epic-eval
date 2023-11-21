//
// Created by Shujian Qian on 2023-11-20.
//
#include <util_gpu_error_check.cuh>

namespace epic {

void *allocatePinnedMemory(size_t size)
{
    void *retval = nullptr;
    gpu_err_check(cudaMallocHost(&retval, size, cudaHostAllocDefault));
    return retval;
}

void freePinedMemory(void *ptr)
{
    gpu_err_check(cudaFreeHost(ptr));
}

} // namespace epic