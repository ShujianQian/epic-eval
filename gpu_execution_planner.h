//
// Created by Shujian Qian on 2023-08-09.
//

#ifndef GPU_EXECUTION_PLANNER_H
#define GPU_EXECUTION_PLANNER_H

#include <any>
#include <cstdint>
#include <string_view>

#include "execution_planner.h"
#include "allocator.h"
#include "base_record.h"

#ifdef EPIC_CUDA_AVAILABLE

namespace epic {

class GpuTableExecutionPlanner : public TableExecutionPlanner
{
public:
    const std::string_view name;
    const size_t record_size;
    Allocator &allocator;
    const size_t max_ops_per_txn;
    const size_t max_num_txns;
    const size_t max_num_records;
    const size_t max_num_ops;
    //    void *d_records = nullptr;
    //    void *d_temp_versions = nullptr;
    //    uint32_t *d_num_ops = nullptr;
    //    uint32_t *d_op_offsets = nullptr;
    //    void *d_submitted_ops = nullptr;
    void *d_sorted_ops = nullptr;
    void *d_write_ops_before = nullptr;
    void *d_write_ops_after = nullptr;
    void *d_rw_ops_type = nullptr;
    void *d_tver_write_ops_before = nullptr;
    void *d_rw_locations = nullptr;
    void *d_copy_dep_locations = nullptr;

    void *d_output_txn_array;
    size_t output_txn_array_baseTxn_size;

    std::any cuda_stream;
    //    void *d_scratch_array = nullptr; /* for cub sort and scan */

    template<typename TxnType>
    GpuTableExecutionPlanner(std::string_view name, Allocator &allocator, size_t record_size, size_t max_ops_per_txn,
        size_t max_num_txns, size_t max_num_records, TxnArray<TxnType> &txn_array)
        : name(name)
        , record_size(record_size)
        , allocator(allocator)
        , max_ops_per_txn(max_ops_per_txn)
        , max_num_txns(max_num_txns + 1)
        , max_num_records(max_num_records)
        , max_num_ops(max_num_txns * max_ops_per_txn)
        , output_txn_array_baseTxn_size(TxnArray<TxnType>::kBaseTxnSize)
        , d_output_txn_array(txn_array.txns){};
    virtual ~GpuTableExecutionPlanner() = default;
    void Initialize() override;
    void SubmitOps(CalcNumOpsFunc pre_submit_ops_func, SubmitOpsFunc submit_ops_func) override;
    void InitializeExecutionPlan() override;
    void FinishInitialization() override;
    void ScatterOpLocations() override;
};
} // namespace epic

#endif // EPIC_CUDA_AVAILABLE

#endif // GPU_EXECUTION_PLANNER_H
