//
// Created by Shujian Qian on 2023-11-02.
//

#ifndef EPIC_GACCO_GPU_EXECUTION_PLANNER_H
#define EPIC_GACCO_GPU_EXECUTION_PLANNER_H

#include <cstdint>
#include <cstddef>
#include <string_view>
#include <any>

#include <allocator.h>
#include <gacco/execution_planner.h>
#include <txn.h>

#ifdef EPIC_CUDA_AVAILABLE

namespace gacco {

using epic::Allocator;
using epic::TxnArray;

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
    op_t *d_sorted_ops = nullptr;
    uint32_t *d_unique_rows = nullptr;
    uint32_t *d_unique_access = nullptr;
    uint32_t *d_unique_offset = nullptr;
    uint32_t *d_num_unique_rows = nullptr;

    std::any cuda_stream;

    GpuTableExecutionPlanner(std::string_view name, Allocator &allocator, size_t record_size, size_t max_ops_per_txn,
        size_t max_num_txns, size_t max_num_records);
    ~GpuTableExecutionPlanner() override;
    void Initialize() override;
    void InitializeExecutionPlan() override;
    void FinishInitialization() override;
};

} // namespace gacco

#endif // EPIC_CUDA_AVAILABLE

#endif // EPIC_GACCO_GPU_EXECUTION_PLANNER_H
