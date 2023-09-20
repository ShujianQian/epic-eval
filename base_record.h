//
// Created by Shujian Qian on 2023-08-09.
//

#ifndef BASE_RECORD_H
#define BASE_RECORD_H

#include <cstdint>

namespace epic {
struct BaseRecord
{
    uint32_t epoch_id;
    uint32_t flags;
    uint8_t data[];
};
} // namespace epic

#endif // BASE_RECORD_H
