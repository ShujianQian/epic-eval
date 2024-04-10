//
// Created by Shujian Qian on 2024-04-01.
//

#ifndef UTIL_ENDIANNESS_H
#define UTIL_ENDIANNESS_H

#include <type_traits>
#include <cstdint>

namespace epic {
namespace detail {
template<typename UintType, size_t Sz>
struct endian_swap_impl
{};

template<typename UintType>
struct endian_swap_impl<UintType, 2>
{
    constexpr inline UintType operator()(UintType x)
    {
        return __builtin_bswap16(x);
    }
};

template<typename UintType>
struct endian_swap_impl<UintType, 4>
{
    constexpr inline UintType operator()(UintType x)
    {
        return __builtin_bswap32(x);
    }
};

template<typename UintType>
struct endian_swap_impl<UintType, 8>
{
    constexpr inline UintType operator()(UintType x)
    {
        return __builtin_bswap64(x);
    }
};

} // namespace detail

template<typename UintType>
constexpr inline UintType endian_swap(UintType x)
{
    static_assert(std::is_integral<UintType>::value, "UintType must be an integral type");
    return detail::endian_swap_impl<UintType, sizeof(UintType)>()(x);
}

static_assert(endian_swap(static_cast<uint16_t>(0x1122u)) == 0x2211u, "16bit endian_swap failed");
static_assert(endian_swap(static_cast<uint32_t>(0x11223344u)) == 0x44332211, "32bit endian_swap failed");
static_assert(
    endian_swap(static_cast<uint64_t>(0x1122334455667788ull)) == 0x8877665544332211ull, "64bit endian_swap failed");
} // namespace epic

#endif // UTIL_ENDIANNESS_H
