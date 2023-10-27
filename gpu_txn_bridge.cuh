//
// Created by Shujian Qian on 2023-10-02.
//

#ifndef GPU_TXN_BRIDGE_CUH
#define GPU_TXN_BRIDGE_CUH

#ifdef EPIC_CUDA_AVAILABLE

#    include "txn_bridge.h"

namespace epic {

// class GpuTxnBridge : public TxnBridge {
//     cudaStream_t copy_stream;
// public:
//     GpuTxnBridge() {
//         cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking);
//     }
//     ~GpuTxnBridge() {
//         cudaStreamDestroy(copy_stream);
//     }
//     void StartTransfer(TxnBridgeStorage &storage) override;
//     void FinishTransfer(TxnBridgeStorage &storage) override;
// };
//
}

#endif // EPIC_CUDA_AVAILABLE

#endif // GPU_TXN_BRIDGE_CUH
