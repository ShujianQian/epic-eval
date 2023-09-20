//
// Created by Shujian Qian on 2023-08-19.
//

#ifndef TXN_H
#define TXN_H

#include <cstdint>
#include <cstdlib>
#include <cassert>

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
class TxnArray
{
public:
    static constexpr size_t kBaseTxnSize = BaseTxnSize<Txn>::value;
    void *txns = nullptr;
    size_t num_txns;
    size_t epochs;
    TxnArray(size_t num_txns, size_t epochs)
        : num_txns(num_txns)
        , epochs(epochs)
    {
        txns = malloc(kBaseTxnSize * num_txns * epochs);
    }
    ~TxnArray()
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

class TxnRunner
{
public:
    void run(BaseTxn *txn);
};

} // namespace epic

#endif // TXN_H
