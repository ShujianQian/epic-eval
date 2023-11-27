//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_SUBMITTER_H
#define EPIC_BENCHMARKS_YCSB_SUBMITTER_H

#include <cstdint>
#include <cstddef>

#include <txn.h>
#include <benchmarks/ycsb_txn.h>
#include <benchmarks/ycsb_config.h>
#include <util_log.h>

namespace epic::ycsb {

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

    YcsbConfig config;
    TableSubmitDest submit_dest;
    YcsbSubmitter(TableSubmitDest submit_dest, YcsbConfig config)
        : submit_dest(submit_dest)
        , config(config)
    {}
    virtual ~YcsbSubmitter() = default;

    virtual void submit(TxnArray<YcsbTxnParam> &txn_array)
    {
        throw std::runtime_error("YcsbSubmitter::submit not implemented");
    };
};
} // namespace epic::ycsb

#endif // EPIC_BENCHMARKS_YCSB_SUBMITTER_H
