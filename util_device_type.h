//
// Created by Shujian Qian on 2023-10-03.
//

#ifndef UTIL_DEVICE_TYPE_H
#define UTIL_DEVICE_TYPE_H

#include <cstdint>

namespace epic {

enum class DeviceType : uint32_t
{
    CPU = 0,
    GPU
};

}

#endif // UTIL_DEVICE_TYPE_H
