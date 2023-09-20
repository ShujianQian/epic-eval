//
// Created by Shujian Qian on 2023-08-09.
//

#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <cstdlib>

namespace epic {
class Allocator
{
public:
    virtual void *Allocate(size_t size) = 0;
    virtual void Free(void *ptr) = 0;
};
} // namespace epic

#endif // ALLOCATOR_H
