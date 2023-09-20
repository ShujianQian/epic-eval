//
// Created by Shujian Qian on 2023-08-28.
//

#include "util_bitfield.h"

static_assert(epic::detail::FloorLog2<8>::value == 3);
static_assert(epic::detail::FloorLog2<9>::value == 3);
static_assert(epic::detail::CeilLog2<8>::value == 3);
static_assert(epic::detail::CeilLog2<9>::value == 4);

static_assert(epic::detail::SumCeilLog2<8, 9, 10>::value == 11);
static_assert(epic::detail::SumCeilLog2<7>::value == 3);

static_assert(std::is_same<epic::ChooseBitfieldBaseType<6, 7, 8>::type, uint16_t>::value);
static_assert(std::is_same<epic::ChooseBitfieldBaseType<6, 7>::type, uint8_t>::value);
static_assert(std::is_same<epic::ChooseBitfieldBaseType<10'000'000>::type, uint32_t>::value);
static_assert(std::is_same<epic::ChooseBitfieldBaseType<10'000'000, 20, 2 * 128>::type, uint64_t>::value);

int main(int argc, char **argv)
{
    return 0;
}