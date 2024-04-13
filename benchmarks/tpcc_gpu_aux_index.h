//
// Created by Shujian Qian on 2024-04-12.
//

#ifndef TPCC_GPU_AUX_INDEX_H
#define TPCC_GPU_AUX_INDEX_H

#include <any>

#include <txn.h>
#include <benchmarks/tpcc_txn.h>
#include <benchmarks/tpcc_config.h>

namespace epic::tpcc {

class TpccGpuAuxIndex
{
    std::any impl;

public:
    explicit TpccGpuAuxIndex(TpccConfig &config);
    void loadInitialData();
    void insertTxnUpdates(TxnArray<TpccTxn> &txns, size_t epoch);
    void performRangeQueries(TxnArray<TpccTxn> &txns, TxnArray<TpccTxnParam> &index, size_t epoch);
};

} // namespace epic::tpcc

#endif // TPCC_GPU_AUX_INDEX_H
