//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_GPU_INDEX_H
#define EPIC_BENCHMARKS_YCSB_GPU_INDEX_H

#include <any>

#include <benchmarks/ycsb_index.h>
#include <benchmarks/ycsb_config.h>

namespace epic::ycsb {

class YcsbGpuIndex : public YcsbIndex
{
public:
    YcsbConfig ycsb_config;
    std::any gpu_index_impl;
    explicit YcsbGpuIndex(YcsbConfig ycsb_config);

    void loadInitialData() override;
    void indexTxns(TxnArray<YcsbTxn> &txn_array, TxnArray<YcsbTxnParam> &index_array, uint32_t epoch_id) override;
};

} // namespace epic::ycsb

#endif // EPIC_BENCHMARKS_YCSB_GPU_INDEX_H
