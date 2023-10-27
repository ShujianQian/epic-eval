//
// Created by Shujian Qian on 2023-09-20.
//

#ifdef EPIC_CUDA_AVAILABLE

#    include "gpu_txn_bridge.cuh"
#    include "util_log.h"

#    include <cstdio>

namespace epic {
//
// void GpuTxnBridge::StartTransfer(TxnBridgeStorage &storage)
//{
//    if (storage.src_type == DeviceType::CPU && storage.dst_type == DeviceType::GPU) {
//        cudaMemcpyAsync(storage.dest_ptr, storage.src_ptr, storage.txn_size * storage.num_txns, cudaMemcpyHostToDevice, copy_stream);
//    } else if (storage.src_type == DeviceType::GPU && storage.dst_type == DeviceType::CPU) {
//        cudaMemcpyAsync(storage.dest_ptr, storage.src_ptr, storage.txn_size * storage.num_txns, cudaMemcpyDeviceToHost, copy_stream);
//    } else if (storage.src_type == DeviceType::GPU && storage.dst_type == DeviceType::GPU) {
//        return;
//    } else {
//        auto &logger = Logger::GetInstance();
//        logger.Warn("Unsupported transfer on Host using GpuTxnBridge::StartTransfer()");
//        TxnBridge::StartTransfer(storage);
//    }
//}
//
// void GpuTxnBridge::FinishTransfer(TxnBridgeStorage &storage)
//{
//    cudaStreamSynchronize(copy_stream);
//}
//
}

#endif // EPIC_CUDA_AVAILABLE
