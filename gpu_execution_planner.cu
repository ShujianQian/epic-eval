//
// Created by Shujian Qian on 2023-08-13.
//

#include "gpu_execution_planner.cuh"

#include "util_math.h"
#include "util_log.h"
#include "util_arch.h"

namespace epic {

namespace {

size_t allocateDeviceArray(Allocator &allocator, void *&ptr, size_t size, std::string_view name, size_t &total_allocated_size)
{
    epic::Logger &logger = epic::Logger::GetInstance();
    size = AlignTo(size, kDeviceCacheLineSize);
    logger.Trace("Allocating {} bytes for {}", formatSizeBytes(size), name);
    ptr = allocator.Allocate(size);
    total_allocated_size += size;
    return size;
}

} // namespace

void GpuExecutionPlanner::Initialize()
{
    epic::Logger &logger = epic::Logger::GetInstance();
    size_t total_allocated_size = 0;

    logger.Info("Initializing GPU table {}", name);

    allocateDeviceArray(allocator, d_records, max_num_records * record_size, "records", total_allocated_size);
    allocateDeviceArray(allocator, d_temp_versions, max_num_ops * record_size, "temp versions", total_allocated_size);
    allocateDeviceArray(allocator, d_submitted_ops, max_num_ops * sizeof(op_t), "submitted ops", total_allocated_size);
    allocateDeviceArray(allocator, d_sorted_ops, max_num_ops * sizeof(op_t), "sorted ops", total_allocated_size);
    allocateDeviceArray(allocator, d_write_ops_before, max_num_ops * sizeof(uint32_t), "write ops before", total_allocated_size);
    allocateDeviceArray(allocator, d_write_ops_after, max_num_ops * sizeof(uint32_t), "write ops after", total_allocated_size);
    allocateDeviceArray(allocator, d_rw_ops_type, max_num_ops * sizeof(uint8_t), "rw ops type", total_allocated_size);
    allocateDeviceArray(allocator, d_tver_write_ops_before, max_num_ops * sizeof(uint32_t), "temp write ops before", total_allocated_size);
    allocateDeviceArray(allocator, d_rw_locations, max_num_ops * sizeof(uint32_t), "rw locations", total_allocated_size);
    allocateDeviceArray(allocator, d_copy_dep_locations, max_num_ops * sizeof(uint32_t), "copy dep locations", total_allocated_size);
    allocateDeviceArray(allocator, d_scratch_array, 3 * max_num_ops * sizeof(op_t), "scratch array", total_allocated_size);

    logger.Info("Total allocated size for {}: {}", name, formatSizeBytes(total_allocated_size));
}

void GpuExecutionPlanner::SubmitOps(CalcNumOpsFunc pre_submit_ops_func, SubmitOpsFunc submit_ops_func) {}

void GpuExecutionPlanner::InitializeExecutionPlan() {}

void GpuExecutionPlanner::ScatterOpLocations() {}
} // namespace epic