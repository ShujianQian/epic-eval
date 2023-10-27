//
// Created by Shujian Qian on 2023-10-03.
//

#include "gpu_txn.h"
#include "gpu_txn.cuh"
#include "util_log.h"
#include "util_gpu_error_check.cuh"
#include "util_math.h"

namespace epic {

void *createGpuTxnArrayStorage(size_t size)
{
    auto &logger = Logger::GetInstance();
    logger.Trace("Allocating {} bytes for GPU txn array", formatSizeBytes(size));
    void *ptr = nullptr;
    gpu_err_check(cudaMalloc(&ptr, size));
    return ptr;
}

void *destroyGpuTxnArrayStorage(void *ptr)
{
    gpu_err_check(cudaFree(ptr));
    return nullptr;
}

} // namespace epic