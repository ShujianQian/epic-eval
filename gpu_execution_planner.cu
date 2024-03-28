//
// Created by Shujian Qian on 2023-08-13.
//

#include "gpu_execution_planner.h"

#include <cub/cub.cuh>
#include <thrust/functional.h>

#include "util_math.h"
#include "util_log.h"
#include "util_arch.h"
#include "util_gpu_error_check.cuh"
#include "util_cub_reverse_iterator.cuh"
#include <util_gpu.cuh>

namespace epic {

namespace {

size_t allocateDeviceArray(
    Allocator &allocator, void *&ptr, size_t size, std::string_view name, size_t &total_allocated_size)
{
    epic::Logger &logger = epic::Logger::GetInstance();
    size = AlignTo(size, kDeviceCacheLineSize);
    logger.Trace("Allocating {} bytes for {}", formatSizeBytes(size), name);
    ptr = allocator.Allocate(size);
    total_allocated_size += size;
    return size;
}

} // namespace

void GpuTableExecutionPlanner::Initialize()
{
    epic::Logger &logger = epic::Logger::GetInstance();
    size_t total_allocated_size = 0;

    logger.Info("Initializing GPU table {}", name);

    //    allocateDeviceArray(allocator, d_records, max_num_records * record_size, "records", total_allocated_size);
    //    allocateDeviceArray(allocator, d_temp_versions, max_num_ops * record_size, "temp versions",
    //    total_allocated_size);
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_num_ops), max_num_txns * sizeof(uint32_t), "num ops",
        total_allocated_size);
    allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_op_offsets), max_num_txns * sizeof(uint32_t),
        "op offsets", total_allocated_size);
    cudaMemset(d_op_offsets, 0, max_num_txns * sizeof(uint32_t));
    allocateDeviceArray(allocator, d_submitted_ops, max_num_ops * sizeof(op_t), "submitted ops", total_allocated_size);
    allocateDeviceArray(allocator, d_sorted_ops, max_num_ops * sizeof(op_t), "sorted ops", total_allocated_size);
    allocateDeviceArray(
        allocator, d_write_ops_before, max_num_ops * sizeof(uint32_t), "write ops before", total_allocated_size);
    allocateDeviceArray(
        allocator, d_write_ops_after, max_num_ops * sizeof(uint32_t), "write ops after", total_allocated_size);
    allocateDeviceArray(allocator, d_rw_ops_type, max_num_ops * sizeof(uint8_t), "rw ops type", total_allocated_size);
    allocateDeviceArray(allocator, d_tver_write_ops_before, max_num_ops * sizeof(uint32_t), "temp write ops before",
        total_allocated_size);
    allocateDeviceArray(
        allocator, d_rw_locations, max_num_ops * sizeof(uint32_t), "rw locations", total_allocated_size);
    allocateDeviceArray(
        allocator, d_copy_dep_locations, max_num_ops * sizeof(uint32_t), "copy dep locations", total_allocated_size);
    allocateDeviceArray(
        allocator, d_copy_dep_locations, max_num_txns * sizeof(uint32_t *), "txn base pointer", total_allocated_size);
    scratch_array_bytes = std::max(4 * max_num_ops * sizeof(op_t), 2048lu);
    allocateDeviceArray(allocator, d_scratch_array, scratch_array_bytes, "scratch array", total_allocated_size);

    if (pver_type == PVerType::SINGLE_VER_COPY)
    {
        allocateDeviceArray(allocator, d_permenant_version_rids, max_num_ops * sizeof(uint64_t), "pver copy array",
            total_allocated_size);
        allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_num_copy_pver), sizeof(uint32_t), "num copy pver",
            total_allocated_size);
    }

    if (pver_type == PVerType::SINGLE_VER_SYNC)
    {
        const size_t sync_arr_size = max_num_records * sizeof(uint32_t);
        allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_pver_sync_expect), sync_arr_size,
            "pver sync expected array", total_allocated_size);
        allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_pver_sync_counter), sync_arr_size,
            "pver sync counter array", total_allocated_size);
        allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_num_sync_expected), sizeof(uint32_t),
            "num pver sync expected", total_allocated_size);
        allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_sync_row_ids), max_num_ops * sizeof(uint32_t),
            "pver sync row ids", total_allocated_size);
        allocateDeviceArray(allocator, reinterpret_cast<void *&>(d_sync_expected_temp), max_num_ops * sizeof(uint32_t),
            "pver sync expected temp", total_allocated_size);
    }

    logger.Info("Total allocated size for {}: {}", name, formatSizeBytes(total_allocated_size));

    cudaStream_t stream;
    gpu_err_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cuda_stream = stream;
}

void GpuTableExecutionPlanner::SubmitOps(CalcNumOpsFunc pre_submit_ops_func, SubmitOpsFunc submit_ops_func) {}

namespace {
struct SameRow
{
    __host__ __device__ __forceinline__ bool operator()(op_t a, op_t b) const
    {
        return GET_RECORD_ID(a) == GET_RECORD_ID(b);
    }
};
struct WriteExtractor
{
    __host__ __device__ __forceinline__ uint32_t operator()(op_t op) const
    {
        return GET_R_W(op) == write_op;
    }
};
struct VersionWriteExtractor
{
    __host__ __device__ __forceinline__ uint32_t operator()(OperationT op) const
    {
        return op == OperationT::VERSION_WRITE;
    }
};

struct ShouldCopySingleVer
{
    __host__ __device__ __forceinline__ bool operator()(OperationT op) const
    {
        return op == OperationT::RECORD_B_WRITE;
    }
};
struct RowIdExtractor
{
    __host__ __device__ __forceinline__ uint32_t operator()(op_t op) const
    {
        return GET_RECORD_ID(op);
    }
};
struct IsRecordARead
{
    __host__ __device__ __forceinline__ uint32_t operator()(OperationT op) const
    {
        return op == OperationT::RECORD_A_READ;
    }
};

template<typename OpInputIt, typename WBeforeInputIt, typename WAfterInputIt, typename OutputIt>
__global__ void calcOperationType(
    OpInputIt op_input_it, WBeforeInputIt w_before, WAfterInputIt w_after, OutputIt output, size_t size)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }
    OperationT retval;
    if (GET_R_W(op_input_it[idx]) == write_op)
    {
        if (w_after[idx] == 0)
        {
            retval = OperationT::RECORD_B_WRITE;
        }
        else
        {
            retval = OperationT::VERSION_WRITE;
        }
    }
    else
    {
        if (w_before[idx] == 0)
        {
            retval = OperationT::RECORD_A_READ;
        }
        else
        {
            if (w_after[idx] == 0)
            {
                retval = OperationT::RECORD_B_READ;
            }
            else
            {
                retval = OperationT::VERSION_READ;
            }
        }
    }

    output[idx] = retval;
}
__global__ void scatterRWLocation(op_t *sorted_ops, OperationT *op_types, uint32_t *ver_writes_before,
    void *initialize_output, uint32_t base_txn_size, uint32_t num_ops)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_ops)
    {
        return;
    }
    uint32_t rw_location;
    switch (op_types[idx])
    {
    case OperationT::RECORD_A_READ:
    case OperationT::RECORD_A_WRITE:
        rw_location = loc_record_a;
        break;
    case OperationT::RECORD_B_READ:
    case OperationT::RECORD_B_WRITE:
        rw_location = loc_record_b;
        break;
    case OperationT::VERSION_READ:
        rw_location = ver_writes_before[idx] - 1;
        break;
    case OperationT::VERSION_WRITE:
        rw_location = ver_writes_before[idx];
        break;
    default:
        assert(false);
    }
    uint32_t txn_id = GET_TXN_ID(sorted_ops[idx]);
    uint32_t offset = GET_OFFSET(sorted_ops[idx]);
    uint32_t *txn_base_ptr = reinterpret_cast<uint32_t *>(
        reinterpret_cast<BaseTxn *>(static_cast<uint8_t *>(initialize_output) + txn_id * base_txn_size)->data);
    txn_base_ptr[offset] = rw_location;
}
__global__ void scatterPverSyncExpected(
    uint32_t *sync_row_ids, uint32_t *sync_expected_temp, uint32_t *d_num_sync_expected, uint32_t *d_pver_sync_expected)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *d_num_sync_expected)
    {
        return;
    }
    uint32_t row_id = sync_row_ids[idx];
    d_pver_sync_expected[row_id] = sync_expected_temp[idx];
};
} // namespace

void GpuTableExecutionPlanner::InitializeExecutionPlan()
{
    epic::Logger &logger = epic::Logger::GetInstance();
    logger.Info("Initializing execution plan for GPU table {}", name);

    if (curr_num_ops == 0)
    {
        return;
    }

#if 0 // DEBUG
    {
        constexpr uint32_t max_print = 100;
        uint32_t num_print = std::min(curr_num_ops, max_print);
        op_t ops[max_print];

        gpu_err_check(cudaMemcpy(ops, d_submitted_ops, sizeof(op_t) * num_print, cudaMemcpyDeviceToHost));
        logger.Info("table[{}] num submitted ops {}", name, curr_num_ops);

        for (int i = 0; i < num_print; i++)
        {
            logger.Info("table[{}] op{}: record[{}] txn[{}] rw[{}] offset[{}]",
                name, i, GET_RECORD_ID(ops[i]), GET_TXN_ID(ops[i]), GET_R_W(ops[i]), GET_OFFSET(ops[i]));
        }
    }
#endif

    gpu_err_check(cub::DeviceRadixSort::SortKeys(d_scratch_array, scratch_array_bytes,
        static_cast<op_t *>(d_submitted_ops), static_cast<op_t *>(d_sorted_ops), curr_num_ops, 32, sizeof(op_t) * 8,
        std::any_cast<cudaStream_t>(cuda_stream)));

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
            logger.Info("table[{}] op{}: record[{}] txn[{}] rw[{}] offset[{}]",
                name, i, GET_RECORD_ID(ops[i]), GET_TXN_ID(ops[i]), GET_R_W(ops[i]), GET_OFFSET(ops[i]));
        }
    }
#endif

    using IsWriteIter = cub::TransformInputIterator<uint32_t, WriteExtractor, op_t *>;
    IsWriteIter w_before_val(static_cast<op_t *>(d_sorted_ops), WriteExtractor());
    gpu_err_check(cub::DeviceScan::ExclusiveSumByKey(d_scratch_array, scratch_array_bytes,
        static_cast<op_t *>(d_sorted_ops), w_before_val, static_cast<uint32_t *>(d_write_ops_before), curr_num_ops,
        SameRow(), std::any_cast<cudaStream_t>(cuda_stream)));

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
            logger.Info("table[{}] op{}: record[{}] txn[{}] rw[{}] offset[{}] ",
                        name, i, GET_RECORD_ID(ops[i]), GET_TXN_ID(ops[i]), GET_R_W(ops[i]), GET_OFFSET(ops[i]));
        }
    }
#endif

    IsWriteIter w_after_val(static_cast<op_t *>(d_sorted_ops) + curr_num_ops - 1, WriteExtractor());
    ReverseIterator<uint32_t, IsWriteIter> rev_write_after_val(w_after_val);
    ReverseIterator<op_t, op_t *> rev_sorted_ops(static_cast<op_t *>(d_sorted_ops) + curr_num_ops - 1);
    ReverseIterator<uint32_t, uint32_t *> rev_write_ops_after(
        static_cast<uint32_t *>(d_write_ops_after) + curr_num_ops - 1);
    gpu_err_check(cub::DeviceScan::ExclusiveSumByKey(d_scratch_array, scratch_array_bytes, rev_sorted_ops,
        rev_write_after_val, rev_write_ops_after, curr_num_ops, SameRow(), std::any_cast<cudaStream_t>(cuda_stream)));

    calcOperationType<<<(curr_num_ops + 255) / 256, 256, 0, std::any_cast<cudaStream_t>(cuda_stream)>>>(
        static_cast<op_t *>(d_sorted_ops), static_cast<uint32_t *>(d_write_ops_before),
        static_cast<uint32_t *>(d_write_ops_after), static_cast<OperationT *>(d_rw_ops_type), curr_num_ops);

    gpu_err_check(cudaGetLastError());

    using IsVersionWriteIter = cub::TransformInputIterator<uint32_t, VersionWriteExtractor, OperationT *>;
    IsVersionWriteIter version_write_val(static_cast<OperationT *>(d_rw_ops_type), VersionWriteExtractor());
    gpu_err_check(cub::DeviceScan::ExclusiveSum(d_scratch_array, scratch_array_bytes, version_write_val,
        static_cast<uint32_t *>(d_tver_write_ops_before), curr_num_ops, std::any_cast<cudaStream_t>(cuda_stream)));

    scatterRWLocation<<<(curr_num_ops + 255) / 256, 256, 0, std::any_cast<cudaStream_t>(cuda_stream)>>>(
        static_cast<op_t *>(d_sorted_ops), static_cast<OperationT *>(d_rw_ops_type),
        static_cast<uint32_t *>(d_tver_write_ops_before), d_output_txn_array, output_txn_array_baseTxn_size,
        curr_num_ops);
    gpu_err_check(cudaGetLastError());

    if (pver_type == PVerType::SINGLE_VER_COPY)
    {
        using ShouldCopySingleVerIter = cub::TransformInputIterator<bool, ShouldCopySingleVer, OperationT *>;
        ShouldCopySingleVerIter should_copy_single_val(static_cast<OperationT *>(d_rw_ops_type), ShouldCopySingleVer());
        gpu_err_check(cub::DeviceSelect::Flagged(d_scratch_array, scratch_array_bytes,
            static_cast<op_t *>(d_sorted_ops), should_copy_single_val, static_cast<op_t *>(d_permenant_version_rids),
            d_num_copy_pver, curr_num_ops, std::any_cast<cudaStream_t>(cuda_stream)));
        gpu_err_check(cudaMemcpyAsync(&h_num_copy_pver, d_num_copy_pver, sizeof(uint32_t), cudaMemcpyDeviceToHost,
            std::any_cast<cudaStream_t>(cuda_stream)));
    }

    if (pver_type == PVerType::SINGLE_VER_SYNC)
    {
        using RowIdIter = cub::TransformInputIterator<uint32_t, RowIdExtractor, op_t *>;
        RowIdIter row_id_val(static_cast<op_t *>(d_sorted_ops), RowIdExtractor());
        using IsRecordAReadIter = cub::TransformInputIterator<uint32_t, IsRecordARead, OperationT *>;
        IsRecordAReadIter is_record_a_read_val(static_cast<OperationT *>(d_rw_ops_type), IsRecordARead());
        thrust::plus<uint32_t> plus_op;
        gpu_err_check(cub::DeviceReduce::ReduceByKey(d_scratch_array, scratch_array_bytes, row_id_val, d_sync_row_ids,
            is_record_a_read_val, d_sync_expected_temp, d_num_sync_expected, plus_op, curr_num_ops,
            std::any_cast<cudaStream_t>(cuda_stream)));
        gpu_err_check(cudaMemsetAsync(reinterpret_cast<void *>(d_pver_sync_counter), 0,
            max_num_records * sizeof(uint32_t), std::any_cast<cudaStream_t>(cuda_stream)));
        gpu_err_check(cudaMemsetAsync(reinterpret_cast<void *>(d_pver_sync_expect), 0,
            max_num_records * sizeof(uint32_t), std::any_cast<cudaStream_t>(cuda_stream)));
        const uint32_t block_size = 512;
        const uint32_t grid_size = (max_num_ops + block_size - 1) / block_size;
        scatterPverSyncExpected<<<grid_size, block_size, 0, std::any_cast<cudaStream_t>(cuda_stream)>>>(
            d_sync_row_ids, d_sync_expected_temp, d_num_sync_expected, d_pver_sync_expect);
    }
}

void GpuTableExecutionPlanner::FinishInitialization()
{
    if (curr_num_ops == 0)
    {
        return;
    }

    auto &logger = Logger::GetInstance();
    gpu_err_check(cudaStreamSynchronize(std::any_cast<cudaStream_t>(cuda_stream)));

#if 0 // DEBUG
    {
        constexpr uint32_t max_print = 100;
        uint32_t num_print = std::min(curr_num_ops, max_print);
        op_t ops[max_print];
        uint32_t writes_before[max_print], writes_after[max_print];
        OperationT rw_ops_type[max_print];
        uint32_t ver_writes_before[max_print];

        gpu_err_check(cudaMemcpy(ops, d_sorted_ops, sizeof(op_t) * num_print, cudaMemcpyDeviceToHost));
        gpu_err_check(
            cudaMemcpy(writes_before, d_write_ops_before, sizeof(uint32_t) * num_print, cudaMemcpyDeviceToHost));
        gpu_err_check(
            cudaMemcpy(writes_after, d_write_ops_after, sizeof(uint32_t) * num_print, cudaMemcpyDeviceToHost));
        gpu_err_check(cudaMemcpy(rw_ops_type, d_rw_ops_type, sizeof(OperationT) * num_print, cudaMemcpyDeviceToHost));
        gpu_err_check(cudaMemcpy(
            ver_writes_before, d_tver_write_ops_before, sizeof(uint32_t) * num_print, cudaMemcpyDeviceToHost));
        logger.Info("table[{}] num sorted ops {}", name, curr_num_ops);

        auto printOpType = [](OperationT op) -> std::string {
            switch (op)
            {
            case OperationT::VERSION_READ:
                return "VERSION_READ";
            case OperationT::VERSION_WRITE:
                return "VERSION_WRITE";
            case OperationT::RECORD_A_READ:
                return "RECORD_A_READ";
            case OperationT::RECORD_B_READ:
                return "RECORD_B_READ";
            case OperationT::RECORD_A_WRITE:
                return "RECORD_A_WRITE";
            case OperationT::RECORD_B_WRITE:
                return "RECORD_B_WRITE";
            default:
                return "UNKNOWN";
            }
        };
        for (int i = 0; i < num_print; i++)
        {
            logger.Info("table[{}] op{}: record[{}] txn[{}] rw[{}] offset[{}] writes_before[{}] writes_after[{}] "
                        "op[{}] vwrite_before[{}]",
                name, i, GET_RECORD_ID(ops[i]), GET_TXN_ID(ops[i]), GET_R_W(ops[i]), GET_OFFSET(ops[i]),
                writes_before[i], writes_after[i], printOpType(rw_ops_type[i]), ver_writes_before[i]);
        }
    }
#endif

    if (pver_type == PVerType::SINGLE_VER_COPY)
    {
#if 0 // DEBUG
        {
            uint32_t debug_num_copy_pver = std::min(h_num_copy_pver, 100u);
            op_t ops_to_copy[debug_num_copy_pver];
            gpu_err_check(cudaMemcpy(ops_to_copy, d_permenant_version_rids, sizeof(op_t) * debug_num_copy_pver,
                cudaMemcpyDeviceToHost));
            for (int i = 0; i < debug_num_copy_pver; i++)
            {
                logger.Info("table[{}] op{}: record[{}] txn[{}] rw[{}] offset[{}]",
                    name, i, GET_RECORD_ID(ops_to_copy[i]), GET_TXN_ID(ops_to_copy[i]), GET_R_W(ops_to_copy[i]),
                    GET_OFFSET(ops_to_copy[i]));
            }
        }
#endif
        logger.Info("table[{}] num copy pver {}", name, h_num_copy_pver);
    }

    if (pver_type == PVerType::SINGLE_VER_SYNC)
    {
#if 0 // DEBUG
        uint32_t debug_num_pver_sync = 100u;
        uint32_t sync_expected[debug_num_pver_sync];
        gpu_err_check(cudaMemcpy(
            sync_expected, d_pver_sync_expect, sizeof(uint32_t) * debug_num_pver_sync, cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < debug_num_pver_sync; i++)
        {
            logger.Info("table[{}] sync expected for record[{}] is {}", name, i, sync_expected[i]);
        }
#endif

#if 1 // SYNC_STAT
        {
            OperationT *h_rw_ops_type = new OperationT[curr_num_ops];
            gpu_err_check(
                cudaMemcpy(h_rw_ops_type, d_rw_ops_type, sizeof(OperationT) * curr_num_ops, cudaMemcpyDeviceToHost));

            uint32_t num_record_b_writes = 0;
            uint32_t num_record_a_reads = 0;

            for (uint32_t i = 0; i < curr_num_ops; i++)
            {
                if (h_rw_ops_type[i] == OperationT::RECORD_B_WRITE)
                {
                    num_record_b_writes++;
                }
                if (h_rw_ops_type[i] == OperationT::RECORD_A_READ)
                {
                    num_record_a_reads++;
                }
            }

            logger.Info("table[{}] num recordB writes {} num recordA reads {}", name, num_record_b_writes, num_record_a_reads);
            delete[] h_rw_ops_type;
        }
#endif
    }
}

void GpuTableExecutionPlanner::ScatterOpLocations() {}
} // namespace epic