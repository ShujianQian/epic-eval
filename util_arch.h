//
// Created by Shujian Qian on 2023-08-13.
//

#ifndef UTIL_ARCH_H
#define UTIL_ARCH_H

#include <cstdint>

namespace epic {
constexpr size_t kHostCacheLineSize = 64;
constexpr size_t kDeviceCacheLineSize = 128;
} // namespace epic

#endif // UTIL_ARCH_H
