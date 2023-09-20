//
// Created by Shujian Qian on 2023-08-13.
//

#ifndef UTIL_MATH_H
#define UTIL_MATH_H

#include <cstdint>
#include <string>

namespace epic {
template<typename T>
inline T AlignTo(T value, T alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

std::string formatSizeBytes(uint64_t bytes);

constexpr size_t floorLog2(size_t x)
{
    return x == 1 ? 0 : 1 + floorLog2(x >> 1);
}

constexpr size_t ceilLog2(size_t x)
{
    return x == 1 ? 0 : floorLog2(x - 1) + 1;
}

} // namespace epic

#endif // UTIL_MATH_H
