//
// Created by Shujian Qian on 2023-10-08.
//

#include "util_gpu_transfer.h"

#include "util_gpu_error_check.cuh"

namespace epic {

std::any createGpuStream()
{
    cudaStream_t stream;
    gpu_err_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    return stream;
}

void destroyGpuStream(std::any &stream)
{
    auto s = std::any_cast<cudaStream_t>(stream);
    gpu_err_check(cudaStreamDestroy(s));
    stream.reset();
}

void transferCpuToGpu(void *dst, const void *src, size_t size, std::any &stream)
{
    auto s = std::any_cast<cudaStream_t>(stream);
    gpu_err_check(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, s));
}

void transferCpuToGpu(void *dst, const void *src, size_t size)
{
    gpu_err_check(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

void transferGpuToCpu(void *dst, const void *src, size_t size, std::any &stream)
{
    auto s = std::any_cast<cudaStream_t>(stream);
    gpu_err_check(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, s));
}

void transferGpuToCpu(void *dst, const void *src, size_t size)
{
    gpu_err_check(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

void syncGpuStream(std::any &stream)
{
    auto s = std::any_cast<cudaStream_t>(stream);
    gpu_err_check(cudaStreamSynchronize(s));
}

} // namespace epic