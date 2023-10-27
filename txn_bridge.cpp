//
// Created by Shujian Qian on 2023-09-20.
//

#include "txn_bridge.h"

#include <cstdio>

#include "util_log.h"

namespace epic {

void TxnBridge::StartTransfer()
{
    if (src_type == DeviceType::CPU && dst_type == DeviceType::CPU)
    {
        return;
    }
#ifdef EPIC_CUDA_AVAILABLE
    else if (src_type == DeviceType::GPU && dst_type == DeviceType::GPU)
    {
        return;
    }
    else if (src_type == DeviceType::GPU)
    {
        transferGpuToCpu(dest_ptr, src_ptr, txn_size * num_txns, copy_stream);
    }
    else if (dst_type == DeviceType::GPU)
    {
        transferCpuToGpu(dest_ptr, src_ptr, txn_size * num_txns, copy_stream);
    }
#endif
    else
    {
        auto &logger = Logger::GetInstance();
        logger.Error("Unsupported transfer using TxnBridge");
    }
}

void TxnBridge::FinishTransfer() {
    if (src_type == DeviceType::CPU && dst_type == DeviceType::CPU)
    {
        return;
    }
#ifdef EPIC_CUDA_AVAILABLE
    else if (src_type == DeviceType::GPU && dst_type == DeviceType::GPU) {
        return;
    }
    else if (src_type == DeviceType::GPU || dst_type == DeviceType::GPU)
    {
        syncGpuStream(copy_stream);
    }
#endif
    else
    {
        auto &logger = Logger::GetInstance();
        logger.Error("Unsupported transfer using TxnBridge");
    }
}

} // namespace epic
