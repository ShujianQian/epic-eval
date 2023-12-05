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

void *allocatePinnedMemory(size_t size);
inline void *Malloc(size_t size)
{
    void *retval = nullptr;
#ifdef EPIC_CUDA_AVAILABLE
    retval = allocatePinnedMemory(size);
#else
    retval = malloc(size);
#endif
    memset(retval, 0, size);
    return retval;
}

void freePinedMemory(void *ptr);
inline void Free(void *ptr)
{
#ifdef EPIC_CUDA_AVAILABLE
    freePinedMemory(ptr);
#else
    free(ptr);
#endif
}

} // namespace epic

#endif // UTIL_MEMORY_H
