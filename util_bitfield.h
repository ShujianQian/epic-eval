//
// Created by Shujian Qian on 2023-08-28.
//

#ifndef UTIL_BITFIELD_H
#define UTIL_BITFIELD_H

#include <cstdint>
#include <type_traits>

#include "common.h"

namespace epic {

template<typename IntegerType, typename BitfieldType>
EPIC_HOST EPIC_DEVICE constexpr IntegerType bitfieldToInt(BitfieldType bitfield)
{
    static_assert(sizeof(IntegerType) == sizeof(BitfieldType), "IntegerType and BitfieldType must have the same size");
    return *reinterpret_cast<IntegerType *>(&bitfield);
}

template<typename BitfieldType, typename IntegerType>
EPIC_HOST EPIC_DEVICE constexpr BitfieldType intToBitfield(IntegerType integer)
{
    static_assert(sizeof(IntegerType) == sizeof(BitfieldType), "IntegerType and BitfieldType must have the same size");
    return *reinterpret_cast<BitfieldType *>(&integer);
}

namespace detail {

// Recursive template to compute the ceilLog2 of an integer
template<size_t N, size_t Current = 0>
struct FloorLog2
{
    static const size_t value = FloorLog2<(N >> 1), Current + 1>::value;
};

template<size_t Current>
struct FloorLog2<1, Current>
{
    static const size_t value = Current;
};

template<size_t N>
struct CeilLog2
{
    static const size_t value = FloorLog2<N - 1>::value + 1;
};

// Recursive template to compute the sum of ceilLog2 of integers
template<size_t First, size_t... Rest>
struct SumCeilLog2
{
    static const size_t value = CeilLog2<First>::value + SumCeilLog2<Rest...>::value;
};

template<size_t First>
struct SumCeilLog2<First>
{
    static const size_t value = CeilLog2<First>::value;
};

} // namespace detail

template<size_t... Sizes>
struct ChooseBitfieldBaseType
{
    static constexpr size_t TotalBits = detail::SumCeilLog2<Sizes...>::value;

    template<size_t N, typename... Types>
    struct TypeSelector;

    template<size_t N>
    struct TypeSelector<N>
    {
        using type = void;
    };

    template<size_t N, typename First, typename... Rest>
    struct TypeSelector<N, First, Rest...>
    {
        using type = typename std::conditional<N <= sizeof(First) * 8, First, typename TypeSelector<N, Rest...>::type>::type;
    };

    //    using type = typename TypeSelector<TotalBits, uint8_t, uint16_t, uint32_t, uint64_t>::type;
    using type = typename TypeSelector<TotalBits, uint32_t, uint64_t>::type;

    static_assert(!std::is_same<type, void>::value, "Bitfield size too large");
};

} // namespace epic

#endif // UTIL_BITFIELD_H
