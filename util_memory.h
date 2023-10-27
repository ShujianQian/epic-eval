//
// Created by Shujian Qian on 2023-09-15.
//

#ifndef UTIL_MEMORY_H
#define UTIL_MEMORY_H

#include <cstdlib>

namespace epic {

struct FreeDelete
{
    void operator()(void *ptr)
    {
        free(ptr);
    }
};

} // namespace epic

#endif // UTIL_MEMORY_H
