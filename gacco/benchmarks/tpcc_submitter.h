//
// Created by Shujian Qian on 2023-11-08.
//

#ifndef EPIC_GACCO_BENCHMARKS_TPCC_SUBMITTER_H
#define EPIC_GACCO_BENCHMARKS_TPCC_SUBMITTER_H

#include <any>
#include <vector>

#include <gacco/gpu_execution_planner.h>
#include <gacco/benchmarks/tpcc_txn.h>
#include <txn.h>
#include <util_log.h>
#include <tpcc_config.h>

#ifdef EPIC_CUDA_AVAILABLE

namespace gacco::tpcc {

using epic::tpcc::TpccConfig;

using epic::TxnArray;

class TpccSubmitter
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

    TableSubmitDest warehouse_submit_dest;
    TableSubmitDest district_submit_dest;
    TableSubmitDest customer_submit_dest;
    TableSubmitDest history_submit_dest;
    TableSubmitDest new_order_submit_dest;
    TableSubmitDest order_submit_dest;
    TableSubmitDest order_line_submit_dest;
    TableSubmitDest item_submit_dest;
    TableSubmitDest stock_submit_dest;
    epic::tpcc::TpccConfig config;

    virtual ~TpccSubmitter() = default;
    TpccSubmitter(TableSubmitDest warehouse_submit_dest, TableSubmitDest district_submit_dest,
        TableSubmitDest customer_submit_dest, TableSubmitDest history_submit_dest,
        TableSubmitDest new_order_submit_dest, TableSubmitDest order_submit_dest,
        TableSubmitDest order_line_submit_dest, TableSubmitDest item_submit_dest, TableSubmitDest stock_submit_dest,
        epic::tpcc::TpccConfig config)
        : warehouse_submit_dest(warehouse_submit_dest)
        , district_submit_dest(district_submit_dest)
        , customer_submit_dest(customer_submit_dest)
        , history_submit_dest(history_submit_dest)
        , new_order_submit_dest(new_order_submit_dest)
        , order_submit_dest(order_submit_dest)
        , order_line_submit_dest(order_line_submit_dest)
        , item_submit_dest(item_submit_dest)
        , stock_submit_dest(stock_submit_dest)
        , config(config)
    {}

    virtual void submit(TxnArray<TpccTxnParam> &txn_array)
    {
        auto &logger = epic::Logger::GetInstance();
        logger.Error("TpccSubmittor::submit not implemented");
    };
};
} // namespace gacco::tpcc

#endif // EPIC_CUDA_AVAILABLE

#endif // EPIC_GACCO_BENCHMARKS_TPCC_SUBMITTER_H
