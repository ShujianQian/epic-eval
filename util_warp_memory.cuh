//
// Created by Shujian Qian on 2023-11-01.
//

#ifndef UTIL_WARP_MEMORY_CUH
#define UTIL_WARP_MEMORY_CUH

#include <util_arch.h>

namespace epic {

__device__ __forceinline__ uint32_t warpMemcpy(uint32_t *dst, const uint32_t *src, uint32_t length, uint32_t lane_id) {
    for (uint32_t offset = 0; offset < length; offset += kDeviceWarpSize) {
        if (offset + lane_id < length) {
            dst[offset + lane_id] = src[offset + lane_id];
        }
    }
}

}

#endif // UTIL_WARP_MEMORY_CUH
