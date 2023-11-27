//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_INDEX_H
#define EPIC_BENCHMARKS_YCSB_INDEX_H

#include <txn.h>
#include <benchmarks/ycsb_txn.h>

namespace epic::ycsb {

class YcsbIndex
{
public:
    virtual void indexTxns(TxnArray<YcsbTxn> &txn_array, TxnArray<YcsbTxnParam> &index_array, uint32_t epoch_id) = 0;
    virtual void loadInitialData() = 0;
    virtual ~YcsbIndex() = default;
};

} // namespace epic::ycsb

#endif // EPIC_BENCHMARKS_YCSB_INDEX_H
