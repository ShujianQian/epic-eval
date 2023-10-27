//
// Created by Shujian Qian on 2023-10-11.
//

#ifndef TPCC_GPU_SUBMITTER_H
#define TPCC_GPU_SUBMITTER_H

#include "benchmarks/tpcc_submitter.h"

#include <any>
#include <vector>

#ifdef EPIC_CUDA_AVAILABLE

namespace epic::tpcc {

class TpccGpuSubmitter : public TpccSubmitter
{
    std::vector<std::any> cuda_streams;

public:
    TpccGpuSubmitter(TableSubmitDest warehouse_submit_dest, TableSubmitDest district_submit_dest,
        TableSubmitDest customer_submit_dest, TableSubmitDest history_submit_dest,
        TableSubmitDest new_order_submit_dest, TableSubmitDest order_submit_dest,
        TableSubmitDest order_line_submit_dest, TableSubmitDest item_submit_dest, TableSubmitDest stock_submit_dest);

    ~TpccGpuSubmitter() override;

    void submit(TxnArray<TpccTxnParam> &txn_array) override;
};

} // namespace epic::tpcc

#endif // EPIC_CUDA_AVAILABLE

#endif // TPCC_GPU_SUBMITTER_H
