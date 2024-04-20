//
// Created by Shujian Qian on 2024-04-20.
//

#ifndef MICRO_STORAGE_H
#define MICRO_STORAGE_H

#include <storage.h>

namespace epic::micro {

struct MicroValue
{
  uint8_t data[100];
};

using MicroVersion = Version<MicroValue>;
using MicroRecord = Record<MicroValue>;

}

#endif  // MICRO_STORAGE_H
