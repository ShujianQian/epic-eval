//
// Created by Shujian Qian on 2023-08-08.
//

#ifndef YCSB_H
#define YCSB_H

#include <cstdint>

#include "txn.h"

namespace epic {
namespace ycsb {

class YcsbRecord
{
public:
    uint8_t data[1000];
};
} // namespace ycsb
} // namespace epic

#endif // YCSB_H
