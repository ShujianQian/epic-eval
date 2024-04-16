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
            /* FIXME: add a relink function */
//            if (dest.txns != nullptr)
//            {
//                dest.Destroy();
//            }
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

class PackedTxnBridge
{
    DeviceType src_type = DeviceType::CPU;
    DeviceType dst_type = DeviceType::CPU;
    size_t txn_size = 0;
    size_t num_txns = 0;
    uint32_t packed_size = 0;
    void *src_ptr = nullptr;
    void *src_index_ptr = nullptr;
    void *dest_ptr = nullptr;
    void *dest_index_ptr = nullptr;
    uint32_t *src_size_ptr = nullptr;
    uint32_t *dest_size_ptr = nullptr;
    std::any copy_stream; /* used for GPU only */
public:
    template<typename TxnType>
    void Link(PackedTxnArray<TxnType> &src, PackedTxnArray<TxnType> &dest)
    {
        auto &logger = Logger::GetInstance();
        if (src.num_txns != dest.num_txns)
        {
            throw std::runtime_error("TxnArray::num_txns mismatch");
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
            /* FIXME: add a relink function */
//            if (dest.txns != nullptr)
//            {
//                dest.Destroy();
//            }
            dest.txns = src.txns;
            dest.index = src.index;
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
        src_index_ptr = src.index;
        dest_index_ptr = dest.index;
        src_size_ptr = &src.size;
        dest_size_ptr = &dest.size;
    }

    virtual ~PackedTxnBridge()
    {
        if (copy_stream.has_value())
        {
            destroyGpuStream(copy_stream);
        }
    }

    virtual void StartTransfer()
    {
        auto &logger = Logger::GetInstance();
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
            transferGpuToCpu(&packed_size, &static_cast<uint32_t *>(src_index_ptr)[num_txns], sizeof(uint32_t));
            transferGpuToCpu(dest_ptr, src_ptr, packed_size, copy_stream);
            transferGpuToCpu(dest_index_ptr, src_index_ptr, num_txns * sizeof(uint32_t), copy_stream);
            *src_size_ptr = packed_size;
            *dest_size_ptr = packed_size;
            logger.Info("Transferred {} bytes from GPU to CPU", packed_size);
        }
        else if (dst_type == DeviceType::GPU)
        {
            packed_size = *src_size_ptr;
            transferCpuToGpu(dest_ptr, src_ptr, packed_size, copy_stream);
            transferCpuToGpu(dest_index_ptr, src_index_ptr, num_txns * sizeof(uint32_t), copy_stream);
            *dest_size_ptr = packed_size;
        }
#endif
        else
        {
            logger.Error("Unsupported transfer using TxnBridge");
        }
    }

    virtual void FinishTransfer()
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
};
} // namespace epic

#endif // TXN_BRIDGE_H
