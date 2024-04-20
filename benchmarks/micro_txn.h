//
// Created by Shujian Qian on 2024-04-20.
//

#ifndef MICRO_TXN_H
#define MICRO_TXN_H

#include <cstdint>

namespace epic::micro {

struct MicroTxn
{
    uint32_t keys[10];
    bool abort;
};

struct MicroTxnParams
{
    uint32_t record_ids[10];
    bool abort;
};

struct MicroTxnExecPlan
{
    uint32_t read_locs[10];
    uint32_t write_locs[10];
};
} // namespace epic::micro

#endif // MICRO_TXN_H
