//
// Created by Shujian Qian on 2023-11-27.
//

#ifndef EPIC_GACCO_BENCHMARKS_YCSB_GPU_SUBMITTER_H
#define EPIC_GACCO_BENCHMARKS_YCSB_GPU_SUBMITTER_H

#include <any>
#include <vector>

#include <gacco/benchmarks/ycsb_submitter.h>
#include <gacco/benchmarks/ycsb_txn.h>
#include <txn.h>

#ifdef EPIC_CUDA_AVAILABLE

namespace gacco::ycsb {

using epic::TxnArray;

class YcsbGpuSubmitter : public YcsbSubmitter
{
    std::any cuda_stream;

public:
    YcsbGpuSubmitter(TableSubmitDest table_submit_dest, YcsbConfig config);
    ~YcsbGpuSubmitter() override;

    void submit(TxnArray<YcsbTxnParam> &txn_array) override;
};

} // namespace gacco::ycsb

#endif // EPIC_CUDA_AVAILABLE

#endif // EPIC_GACCO_BENCHMARKS_YCSB_GPU_SUBMITTER_H
