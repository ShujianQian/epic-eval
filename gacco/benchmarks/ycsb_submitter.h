//
// Created by Shujian Qian on 2023-11-08.
//

#ifndef EPIC_GACCO_BENCHMARKS_YCSB_SUBMITTER_H
#define EPIC_GACCO_BENCHMARKS_YCSB_SUBMITTER_H

#include <any>
#include <vector>

#include <txn.h>
#include <util_log.h>
#include <benchmarks/ycsb_config.h>
#include <gacco/benchmarks/ycsb_txn.h>
#include <gacco/gpu_execution_planner.h>

#ifdef EPIC_CUDA_AVAILABLE

namespace gacco::ycsb {

using epic::TxnArray;
using epic::ycsb::YcsbConfig;

class YcsbSubmitter
{
public:
    struct TableSubmitDest
    {
        uint32_t *d_num_ops = nullptr;
        uint32_t *d_op_offsets = nullptr;
        void *d_submitted_ops = nullptr;
        void *temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        uint32_t &curr_num_ops;
    };

    TableSubmitDest submit_dest;
    YcsbConfig config;

    virtual ~YcsbSubmitter() = default;
    YcsbSubmitter(TableSubmitDest submit_dest, YcsbConfig config)
        : submit_dest(submit_dest)
        , config(config)
    {}

    virtual void submit(TxnArray<YcsbTxnParam> &txn_array)
    {
        throw std::runtime_error("YcsbSubmitter::submit Not implemented");
    };
};
} // namespace gacco::ycsb

#endif // EPIC_CUDA_AVAILABLE

#endif // EPIC_GACCO_BENCHMARKS_YCSB_SUBMITTER_H
