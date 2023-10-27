//
// Created by Shujian Qian on 2023-10-03.
//

#ifndef GPU_TXN_H
#define GPU_TXN_H

#ifdef EPIC_CUDA_AVAILABLE

namespace epic {

void *createGpuTxnArrayStorage(size_t size);

void *destroyGpuTxnArrayStorage(void *ptr);

} // namespace epic

#endif

#endif // GPU_TXN_H
