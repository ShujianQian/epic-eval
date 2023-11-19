//
// Created by Shujian Qian on 2023-11-08.
//

#ifndef EPIC_GACCO_BENCHMARKS_TPCC_GPU_SUBMITTER_H
#define EPIC_GACCO_BENCHMARKS_TPCC_GPU_SUBMITTER_H

#include <any>
#include <vector>

#include <gacco/benchmarks/tpcc_submitter.h>
#include <gacco/benchmarks/tpcc_txn.h>
#include <txn.h>

#ifdef EPIC_CUDA_AVAILABLE

namespace gacco::tpcc {

using epic::TxnArray;

class TpccGpuSubmitter : public TpccSubmitter
{
    std::vector<std::any> cuda_streams;

public:

    TpccGpuSubmitter(TableSubmitDest warehouse_submit_dest, TableSubmitDest district_submit_dest,
        TableSubmitDest customer_submit_dest, TableSubmitDest history_submit_dest,
        TableSubmitDest new_order_submit_dest, TableSubmitDest order_submit_dest,
        TableSubmitDest order_line_submit_dest, TableSubmitDest item_submit_dest, TableSubmitDest stock_submit_dest, TpccConfig config);
    ~TpccGpuSubmitter();

    void submit(TxnArray<TpccTxnParam> &txn_array);
};

} // namespace gacco::tpcc

#endif // EPIC_CUDA_AVAILABLE

#endif // EPIC_GACCO_BENCHMARKS_TPCC_GPU_SUBMITTER_H
