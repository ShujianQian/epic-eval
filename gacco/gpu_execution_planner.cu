//
// Created by Shujian Qian on 2023-11-02.
//

#include <gacco/gpu_execution_planner.h>

#include <util_log.h>
#include <util_arch.h>
#include <util_math.h>
#include <util_cub_temp_arr_size.cuh>
#include <util_gpu_error_check.cuh>

namespace gacco {

namespace {

size_t allocateDeviceArray(
    Allocator &allocator, void *&ptr, size_t size, std::string_view name, size_t &total_allocated_size)
{
    epic::Logger &logger = epic::Logger::GetInstance();
    size = epic::AlignTo(size, epic::kDeviceCacheLineSize);
    logger.Trace("Allocating {} bytes for {}", epic::formatSizeBytes(size), name);
    ptr = allocator.Allocate(size);
    total_allocated_size += size;
    return size;
}

struct RowIdExtractOp
{
    __host__ __device__ __forceinline__ uint32_t operator()(op_t op) const
    {
        return GACCO_GET_RECORD_ID(op);
    }
};

__global__ void copyAccessTidKernel(uint32_t *access, op_t *sorted_ops, uint32_t num_ops)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_ops)
    {
        return;
    }
    access[tid] = GACCO_GET_TXN_ID(sorted_ops[tid]);
}

__global__ void buildAuxArrayKernel(uint32_t *access, uint32_t *offset, uint32_t *lock, uint32_t *unique_offset,
    uint32_t *unique_rows, uint32_t num_unique_rows)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_unique_rows)
    {
        return;
    }
    uint32_t row_id = unique_rows[tid];
    uint32_t my_offset = unique_offset[tid];
    offset[row_id] = my_offset;
    lock[row_id] = access[my_offset];
}

} // namespace

void GpuTableExecutionPlanner::Initialize()
{
    epic::Logger &logger = epic::Logger::GetInstance();
    size_t total_allocated_size = 0;

    logger.Info("Initializing GPU table {}", name);
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_num_ops), max_num_txns * sizeof(uint32_t), "num ops",
        total_allocated_size);
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_op_offsets), max_num_txns * sizeof(uint32_t),
        "op offsets", total_allocated_size);
    cudaMemset(d_op_offsets, 0, max_num_txns * sizeof(uint32_t));
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_submitted_ops), max_num_ops * sizeof(op_t),
        "submitted ops", total_allocated_size);
    scratch_array_bytes = epic::getCubTempArraySize<op_t>(max_num_ops, max_num_records);
    logger.Info("Scratch array size: {}", epic::formatSizeBytes(scratch_array_bytes));
    allocateDeviceArray(allocator, d_scratch_array, scratch_array_bytes, "scratch array", total_allocated_size);

    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_sorted_ops), max_num_ops * sizeof(op_t), "sorted ops",
        total_allocated_size);

    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_unique_rows), max_num_records * sizeof(uint32_t),
        "unique rows", total_allocated_size);
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_unique_access), max_num_records * sizeof(uint32_t),
        "unique rows accesses", total_allocated_size);
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_unique_offset), max_num_records * sizeof(uint32_t),
        "unique rows offset", total_allocated_size);

    allocateDeviceArray(allocator, reinterpret_cast<void *&>(table_lock.access), max_num_ops * sizeof(uint32_t), "access",
        total_allocated_size);
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(table_lock.offset), max_num_records * sizeof(uint32_t), "offset",
        total_allocated_size);
    allocateDeviceArray(
        allocator, reinterpret_cast<void *&>(table_lock.lock), max_num_records * sizeof(uint32_t), "lock", total_allocated_size);
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_num_unique_rows), sizeof(uint32_t), "num unique rows",
        total_allocated_size);

    logger.Info("Total allocated size for {}: {}", name, epic::formatSizeBytes(total_allocated_size));

    cudaStream_t stream;
    gpu_err_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    this->cuda_stream = stream;
}

void GpuTableExecutionPlanner::InitializeExecutionPlan()
{
    epic::Logger &logger = epic::Logger::GetInstance();
    logger.Info("Initializing execution plan for GPU table {}", name);

    if (curr_num_ops == 0)
    {
        return;
    }

    /* only need to sort record id, txn_id is already sorted during submission */
    gpu_err_check(cub::DeviceRadixSort::SortKeys(d_scratch_array, scratch_array_bytes,
        static_cast<op_t *>(d_submitted_ops), static_cast<op_t *>(d_sorted_ops), curr_num_ops, record_id_shift,
        sizeof(op_t) * 8, std::any_cast<cudaStream_t>(cuda_stream)));

#if 0 // DEBUG
    {
        gpu_err_check(cudaStreamSynchronize(std::any_cast<cudaStream_t>(cuda_stream)));
        constexpr uint32_t max_print = 100;
        uint32_t num_print = std::min(curr_num_ops, max_print);
        op_t ops[max_print];

        gpu_err_check(cudaMemcpy(ops, d_sorted_ops, sizeof(op_t) * num_print, cudaMemcpyDeviceToHost));
        logger.Info("table[{}] num submitted ops {}", name, curr_num_ops);

        for (int i = 0; i < num_print; i++)
        {
            logger.Info("table[{}] op{}: record[{}] txn[{}]",
                name, i, GACCO_GET_RECORD_ID(ops[i]), GACCO_GET_TXN_ID(ops[i]));
        }
    }
#endif

    // DEBUG
    //    gpu_err_check(cudaMemsetAsync(d_unique_rows, 0, sizeof(uint32_t), std::any_cast<cudaStream_t>(cuda_stream)));

    using RowIdIter = cub::TransformInputIterator<uint32_t, RowIdExtractOp, op_t *>;
    RowIdIter row_id_iter(static_cast<op_t *>(d_sorted_ops), RowIdExtractOp());
    gpu_err_check(cub::DeviceRunLengthEncode::Encode(d_scratch_array, scratch_array_bytes, row_id_iter, d_unique_rows,
        d_unique_access, d_num_unique_rows, curr_num_ops, std::any_cast<cudaStream_t>(cuda_stream)));

    uint32_t h_num_unique_rows = 0;
    gpu_err_check(cudaMemcpyAsync(&h_num_unique_rows, d_num_unique_rows, sizeof(uint32_t), cudaMemcpyDeviceToHost,
        std::any_cast<cudaStream_t>(cuda_stream)));

    gpu_err_check(cudaStreamSynchronize(std::any_cast<cudaStream_t>(cuda_stream)));

#if 0 // DEBUG
    {
        gpu_err_check(cudaStreamSynchronize(std::any_cast<cudaStream_t>(cuda_stream)));
        constexpr uint32_t max_print = 100;
        uint32_t num_print = std::min(h_num_unique_rows, max_print);
        uint32_t rows[max_print];
        uint32_t access[max_print];

        gpu_err_check(cudaMemcpy(rows, d_unique_rows, sizeof(uint32_t) * num_print, cudaMemcpyDeviceToHost));
        gpu_err_check(cudaMemcpy(access, d_unique_access, sizeof(uint32_t) * num_print, cudaMemcpyDeviceToHost));
        logger.Info("table[{}] num unique rows {}", name, h_num_unique_rows);

        for (int i = 0; i < num_print; i++)
        {
            logger.Info("table[{}]: record[{}] num_access[{}]",
                        name, rows[i], access[i]);
        }
    }
#endif

    gpu_err_check(cub::DeviceScan::ExclusiveSum(d_scratch_array, scratch_array_bytes, d_unique_access, d_unique_offset,
        h_num_unique_rows, std::any_cast<cudaStream_t>(cuda_stream)));

    constexpr uint32_t block_size = 256;
    copyAccessTidKernel<<<(curr_num_ops + block_size - 1) / block_size, block_size, 0,
        std::any_cast<cudaStream_t>(cuda_stream)>>>(table_lock.access, d_sorted_ops, curr_num_ops);
    gpu_err_check(cudaPeekAtLastError());

    buildAuxArrayKernel<<<(h_num_unique_rows + block_size - 1) / block_size, block_size, 0,
        std::any_cast<cudaStream_t>(cuda_stream)>>>(
        table_lock.access, table_lock.offset, table_lock.lock, d_unique_offset, d_unique_rows, h_num_unique_rows);
    gpu_err_check(cudaPeekAtLastError());

    logger.Info("table[{}] num unique rows {}", name, h_num_unique_rows);

#if 0 // DEBUG
    {
        gpu_err_check(cudaStreamSynchronize(std::any_cast<cudaStream_t>(cuda_stream)));
        constexpr uint32_t max_print = 100;
        uint32_t num_print = std::min(h_num_unique_rows, max_print);
        uint32_t lock[max_print];
        uint32_t access[max_print];

        gpu_err_check(cudaMemcpy(lock, table_lock.lock, sizeof(uint32_t) * num_print, cudaMemcpyDeviceToHost));
        gpu_err_check(cudaMemcpy(access, table_lock.access, sizeof(uint32_t) * num_print, cudaMemcpyDeviceToHost));
        logger.Info("table[{}] num unique rows {}", name, h_num_unique_rows);

        for (int i = 0; i < num_print; i++)
        {
            logger.Info("table[{}][{}]: access[{}]",
                        name, i, access[i]);
        }
        for (int i = 0; i < num_print; i++)
        {
            logger.Info("table[{}][{}]: lock[{}]",
                        name, i, lock[i]);
        }
    }
#endif
}

void GpuTableExecutionPlanner::FinishInitialization()
{
    if (curr_num_ops == 0)
    {
        return;
    }

    auto &logger = epic::Logger::GetInstance();
    gpu_err_check(cudaStreamSynchronize(std::any_cast<cudaStream_t>(cuda_stream)));
}

GpuTableExecutionPlanner::GpuTableExecutionPlanner(std::string_view name, Allocator &allocator, size_t record_size,
    size_t max_ops_per_txn, size_t max_num_txns, size_t max_num_records)
    : name(name)
    , record_size(record_size)
    , allocator(allocator)
    , max_ops_per_txn(max_ops_per_txn)
    , max_num_txns(max_num_txns + 1)
    , max_num_records(max_num_records)
    , max_num_ops(max_num_txns * max_ops_per_txn){};

GpuTableExecutionPlanner::~GpuTableExecutionPlanner()
{
    if (cuda_stream.has_value())
    {
        cudaStreamDestroy(std::any_cast<cudaStream_t>(cuda_stream));
    }
}

} // namespace gacco