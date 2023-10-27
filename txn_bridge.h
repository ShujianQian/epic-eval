//
// Created by Shujian Qian on 2023-09-20.
//

#ifndef TXN_BRIDGE_H
#define TXN_BRIDGE_H

#include <any>
#include <cstdint>
#include <cstdio>

#include "common.h"
#include "txn.h"
#include "util_device_type.h"
#include "util_gpu_transfer.h"

namespace epic {

struct TxnBridgeStorage
{
    DeviceType src_type;
    DeviceType dst_type;
    size_t txn_size;
    size_t num_txns;
    uint8_t *src_ptr;
    uint8_t *dest_ptr;
};

class TxnBridge
{
    DeviceType src_type = DeviceType::CPU;
    DeviceType dst_type = DeviceType::CPU;
    size_t txn_size = 0;
    size_t num_txns = 0;
    void *src_ptr = nullptr;
    void *dest_ptr = nullptr;
    std::any copy_stream; /* used for GPU only */
public:
    template<typename TxnType>
    void Link(TxnArray<TxnType> &src, TxnArray<TxnType> &dest)
    {
        auto &logger = Logger::GetInstance();
        if (src.num_txns != dest.num_txns)
        {
            logger.Error("TxnArray::num_txns mismatch");
        }

        src_type = src.device;
        dst_type = dest.device;
        txn_size = BaseTxnSize<TxnType>::value;
        num_txns = src.num_txns;

        if (src.txns == nullptr)
        {
            src.Initialize();
        }

        if (src.device == dest.device)
        {
            if (dest.txns != nullptr)
            {
                dest.Destroy();
            }
            dest.txns = src.txns;
        }
        else
        {
            if (dest.txns == nullptr)
            {
                dest.Initialize();
            }
        }

        if (src.device == DeviceType::GPU || dest.device == DeviceType::GPU)
        {
            if (!copy_stream.has_value())
            {
                copy_stream = createGpuStream();
                logger.Trace("Created a new stream for GPU copy with type {}", copy_stream.type().name());
            }
        }

        src_ptr = src.txns;
        dest_ptr = dest.txns;
    }

    ~TxnBridge()
    {
        if (copy_stream.has_value())
        {
            destroyGpuStream(copy_stream);
        }
    }

    virtual void StartTransfer();
    virtual void FinishTransfer();
};

} // namespace epic

#endif // TXN_BRIDGE_H
