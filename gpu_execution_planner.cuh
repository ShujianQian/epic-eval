//
// Created by Shujian Qian on 2023-08-09.
//

#ifndef GPU_EXECUTION_PLANNER_CUH
#define GPU_EXECUTION_PLANNER_CUH

#include <cstdint>
#include <string_view>
#include <cub/cub.cuh>

#include "execution_planner.h"
#include "allocator.h"
#include "base_record.h"

namespace epic {


class GpuExecutionPlanner : ExecutionPlanner
{
public:
    const std::string_view name;
    const size_t record_size;
    Allocator &allocator;
    const size_t max_ops_per_txn;
    const size_t max_num_txns;
    const size_t max_num_records;
    const size_t max_num_ops;
    void *d_records = nullptr;
    void *d_temp_versions = nullptr;
    void *d_submitted_ops = nullptr;
    void *d_sorted_ops = nullptr;
    void *d_write_ops_before = nullptr;
    void *d_write_ops_after = nullptr;
    void *d_rw_ops_type = nullptr;
    void *d_tver_write_ops_before = nullptr;
    void *d_rw_locations = nullptr;
    void *d_copy_dep_locations = nullptr;
    void *d_scratch_array = nullptr; /* for cub sort and scan */

    GpuExecutionPlanner(std::string_view name, Allocator &allocator, size_t record_size, size_t max_ops_per_txn, size_t max_num_txns,
        size_t max_num_records)
        : name(name)
        , record_size(record_size)
        , allocator(allocator)
        , max_ops_per_txn(max_ops_per_txn)
        , max_num_txns(max_num_txns)
        , max_num_records(max_num_records)
        , max_num_ops(max_num_txns * max_ops_per_txn){};
    virtual ~GpuExecutionPlanner() = default;
    void Initialize() override;
    void SubmitOps(CalcNumOpsFunc pre_submit_ops_func, SubmitOpsFunc submit_ops_func) override;
    void InitializeExecutionPlan() override;
    void ScatterOpLocations() override;

    __device__ __forceinline__ void Write(uint32_t record_id, uint32_t location_id, uint32_t offset, uint64_t data)
    {
        BaseRecord *record = reinterpret_cast<BaseRecord *>(reinterpret_cast<uint8_t *>(d_records) + record_id * record_size);
        reinterpret_cast<uint64_t *>(record->data)[offset] = data;
    };

    __device__ __forceinline__ void WriteMetadata(uint32_t record_id, uint32_t location_id, uint32_t epoch_id, uint32_t flags)
    {
        BaseRecord *record = reinterpret_cast<BaseRecord *>(reinterpret_cast<uint8_t *>(d_records) + record_id * record_size);
        record->flags = flags;
        __syncthreads();
        atomicExch(&record->epoch_id, epoch_id);
    };

    template<typename T>
    __device__ __forceinline__ void SubmitOpsKernel(T *ops_arr, T op, uint32_t txn_id, uint32_t op_id)
    {
        ops_arr[txn_id * max_ops_per_txn + op_id] = op;
    }
};
} // namespace epic

#endif // GPU_EXECUTION_PLANNER_CUH
