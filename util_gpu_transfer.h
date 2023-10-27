//
// Created by Shujian Qian on 2023-10-08.
//

#ifndef UTIL_GPU_TRANSFER_H
#define UTIL_GPU_TRANSFER_H

#include <any>

#ifdef EPIC_CUDA_AVAILABLE

namespace epic {

std::any createGpuStream();

void destroyGpuStream(std::any &stream);

void transferCpuToGpu(void *dst, const void *src, size_t size, std::any &stream);
void transferCpuToGpu(void *dst, const void *src, size_t size);

void transferGpuToCpu(void *dst, const void *src, size_t size, std::any &stream);
void transferGpuToCpu(void *dst, const void *src, size_t size);

void syncGpuStream(std::any &stream);

} // namespace epic

#endif // EPIC_CUDA_AVAILABLE

#endif // UTIL_GPU_TRANSFER_H
