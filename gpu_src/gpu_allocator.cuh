//
// Created by Shujian Qian on 2023-08-09.
//

#ifndef GPU_ALLOCATOR_CUH
#define GPU_ALLOCATOR_CUH

#include "../allocator.h"

namespace epic {
class GpuAllocator : public Allocator
{
public:
    void *Allocate(size_t size) override;
    void Free(void *ptr) override;
};
} // namespace epic

#endif // GPU_ALLOCATOR_CUH
