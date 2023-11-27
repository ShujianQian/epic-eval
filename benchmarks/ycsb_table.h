//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_TABLE_H
#define EPIC_BENCHMARKS_YCSB_TABLE_H

#include <cstdint>

namespace epic::ycsb {

using YcsbKey = uint32_t;

struct YcsbValue
{
    uint8_t data[10][100];
};

struct YcsbFieldValue
{
    uint8_t data[100];
};

}

#endif // EPIC_BENCHMARKS_YCSB_TABLE_H
