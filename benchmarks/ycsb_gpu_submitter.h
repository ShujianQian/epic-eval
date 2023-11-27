//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_GPU_SUBMITTER_H
#define EPIC_BENCHMARKS_YCSB_GPU_SUBMITTER_H

#include <benchmarks/ycsb_submitter.h>

namespace epic::ycsb {

class YcsbGpuSubmitter : public YcsbSubmitter
{
public:
    YcsbGpuSubmitter(TableSubmitDest submit_dest, YcsbConfig config)
        : YcsbSubmitter(submit_dest, config)
    {}

    void submit(TxnArray<YcsbTxnParam> &txn_array) override;
};

} // namespace epic::ycsb

#endif // EPIC_BENCHMARKS_YCSB_GPU_SUBMITTER_H
