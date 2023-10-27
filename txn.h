//
// Created by Shujian Qian on 2023-08-19.
//

#ifndef TXN_H
#define TXN_H

#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <type_traits>

#include "gpu_txn.h"
#include "util_log.h"
#include "util_device_type.h"
#include "util_math.h"

namespace epic {

struct BaseTxn
{
    uint32_t txn_type;
    uint8_t data[];
};

template<typename Txn>
struct BaseTxnSize
{
    static constexpr size_t value = sizeof(BaseTxn) + sizeof(Txn);
};

template<typename Txn>
class TxnInputArray
{
public:
    static constexpr size_t kBaseTxnSize = BaseTxnSize<Txn>::value;
    void *txns = nullptr;
    size_t num_txns;
    size_t epochs;
    TxnInputArray(size_t num_txns, size_t epochs)
        : num_txns(num_txns)
        , epochs(epochs)
    {
        txns = malloc(kBaseTxnSize * num_txns * epochs);
    }
    ~TxnInputArray()
    {
        free(txns);
    }
    inline BaseTxn *getTxn(size_t epoch, size_t index)
    {
        assert(index < num_txns);
        assert(epoch < epochs);
        size_t offset = (epoch * num_txns + index) * kBaseTxnSize;
        return reinterpret_cast<BaseTxn *>(reinterpret_cast<uint8_t *>(txns) + offset);
    }
};

template<typename TxnType>
class TxnArray
{
public:
    static constexpr size_t kBaseTxnSize = BaseTxnSize<TxnType>::value;
    void *txns = nullptr;
    size_t num_txns;
    DeviceType device;

    explicit TxnArray(size_t num_txns)
        : num_txns(num_txns)
        , device(DeviceType::CPU)
    {}

    TxnArray(size_t num_txns, DeviceType device_type, bool initialize = true)
        : num_txns(num_txns)
        , device(device_type)
    {
        if (!initialize)
        {
            return;
        }

        if (device_type == DeviceType::CPU)
        {
            txns = malloc(kBaseTxnSize * num_txns);
        }
#ifdef EPIC_CUDA_AVAILABLE
        else if (device_type == DeviceType::GPU)
        {
            txns = createGpuTxnArrayStorage(kBaseTxnSize * num_txns);
        }
#endif
        else
        {
            auto &logger = Logger::GetInstance();
            logger.Error("Unsupported device type");
            exit(-1);
        }
    }

    void Initialize()
    {
        auto &logger = Logger::GetInstance();
        if (txns != nullptr)
        {
            logger.Error("TxnArray already initialized");
            exit(-1);
        }

        if (device == DeviceType::CPU)
        {
            logger.Trace("Allocating {} bytes for {} txns on CPU", formatSizeBytes(kBaseTxnSize * num_txns), num_txns);
            txns = malloc(kBaseTxnSize * num_txns);
        }
#ifdef EPIC_CUDA_AVAILABLE
        else if (device == DeviceType::GPU)
        {
            logger.Trace("Allocating {} bytes for {} txns on GPU", formatSizeBytes(kBaseTxnSize * num_txns), num_txns);
            txns = createGpuTxnArrayStorage(kBaseTxnSize * num_txns);
        }
#endif // EPIC_CUDA_AVAILABLE
        else
        {
            logger.Error("Unsupported device type");
            exit(-1);
        }
    }

    void Destroy()
    {
        auto &logger = Logger::GetInstance();
        if (txns == nullptr)
        {
            logger.Error("TxnArray already destroyed");
            exit(-1);
        }

        if (device == DeviceType::CPU)
        {
            logger.Trace("Deallocating {} bytes for {} txns on CPU", formatSizeBytes(kBaseTxnSize * num_txns), num_txns);
            free(txns);
            txns = nullptr;
        }
#ifdef EPIC_CUDA_AVAILABLE
        else if (device == DeviceType::GPU)
        {
            logger.Trace("Deallocating {} bytes for {} txns on GPU", formatSizeBytes(kBaseTxnSize * num_txns), num_txns);
            txns = destroyGpuTxnArrayStorage(txns);
        }
#endif
    }

    inline BaseTxn *getTxn(size_t index)
    {
        assert(index < num_txns);
        size_t offset = index * kBaseTxnSize;
        return reinterpret_cast<BaseTxn *>(reinterpret_cast<uint8_t *>(txns) + offset);
    }

    friend class TxnBridge;
};

class TxnRunner
{
public:
    void run(BaseTxn *txn);
};

} // namespace epic

#endif // TXN_H
