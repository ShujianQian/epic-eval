//
// Created by Shujian Qian on 2023-11-22.
//

#include <cmath>
#include <memory>
#include <cuda/std/atomic>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <benchmarks/ycsb_gpu_index.h>
#include <benchmarks/tpcc_gpu_index.h>
#include <benchmarks/tpcc_table.h>
#include <gpu_txn.cuh>
#include <util_log.h>
#include <util_gpu_error_check.cuh>

#include <cuco/static_map.cuh>

#include <cub/cub.cuh>

namespace epic::ycsb {

namespace {

using YcsbIndexType = cuco::static_map<uint32_t, uint32_t>;
using YcsbIndexDeviceView = YcsbIndexType::device_view;

void __global__ prepareYcsbIndexKernel(GpuTxnArray txns, uint32_t *insert, uint32_t num_txns)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns)
    {
        return;
    }
    BaseTxn *base_txn_ptr = txns.getTxn(tid);
    YcsbTxn *txn = reinterpret_cast<YcsbTxn *>(base_txn_ptr->data);
    int base = tid * 10;
    for (int i = 0; i < 10; ++i)
    {
        if (txn->ops[i] == YcsbOpType::INSERT)
        {
            uint32_t key = txn->keys[i];
            insert[base + i] = key;
        }
        else
        {
            insert[base + i] = static_cast<uint32_t>(-1);
        }
    }
}

void __global__ indexYcsbKernel(GpuTxnArray txn, GpuTxnArray index, YcsbIndexDeviceView index_view, uint32_t num_txns)
{

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns)
    {
        return;
    }
    BaseTxn *base_txn_ptr = txn.getTxn(tid);
    BaseTxn *base_index_ptr = index.getTxn(tid);
    YcsbTxn *txn_ptr = reinterpret_cast<YcsbTxn *>(base_txn_ptr->data);
    YcsbTxnParam *index_ptr = reinterpret_cast<YcsbTxnParam *>(base_index_ptr->data);

    for (int i = 0; i < 10; ++i)
    {
        auto record_found = index_view.find(txn_ptr->keys[i]);
        if (record_found != index_view.end())
        {
            index_ptr->record_ids[i] = record_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            printf("record not found\n");
            assert(false);
        }
        index_ptr->ops[i] = txn_ptr->ops[i];
        index_ptr->field_ids[i] = txn_ptr->fields[i];
    }
}

class YcsbGpuIndexImpl
{
public:
    static constexpr double load_factor = 0.5;
//    static constexpr cuco::empty_key<uint32_t> empty_key_sentinel{static_cast<uint32_t>(-1)};
//    static constexpr cuco::empty_value<uint32_t> empty_value_sentinel{static_cast<uint32_t>(-1)};
        static constexpr cuco::empty_key<uint32_t> empty_key_sentinel{0xffffffff};
        static constexpr cuco::empty_value<uint32_t> empty_value_sentinel{0xffffffff};

    YcsbConfig ycsb_config;
    std::shared_ptr<YcsbIndexType> index;
    YcsbIndexDeviceView index_view;

    uint32_t *d_free_rows;
    thrust::device_ptr<uint32_t> dp_free_rows;
    uint32_t free_start = 0;

    uint32_t *d_inserts, *d_valid_inserts;
    thrust::device_ptr<uint32_t> dp_inserts, dp_valid_inserts;
    uint32_t *d_num_insert;

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    explicit YcsbGpuIndexImpl(YcsbConfig ycsb_config)
        : ycsb_config(ycsb_config)
        , index_view(index->get_device_view())
        , index(std::make_shared<YcsbIndexType>(static_cast<size_t>(std::ceil(ycsb_config.num_records / load_factor)),
              empty_key_sentinel, empty_value_sentinel))
    {
        auto &logger = Logger::GetInstance();

        gpu_err_check(cudaMalloc(&d_free_rows, sizeof(uint32_t) * ycsb_config.num_records));
        dp_free_rows = thrust::device_pointer_cast(d_free_rows);

        gpu_err_check(cudaMalloc(&d_inserts, sizeof(uint32_t) * ycsb_config.num_txns * ycsb_config.num_ops_per_txn));
        gpu_err_check(
            cudaMalloc(&d_valid_inserts, sizeof(uint32_t) * ycsb_config.num_txns * ycsb_config.num_ops_per_txn));
        dp_inserts = thrust::device_pointer_cast(d_inserts);
        dp_valid_inserts = thrust::device_pointer_cast(d_valid_inserts);
        gpu_err_check(cudaMalloc(&d_num_insert, sizeof(uint32_t)));

        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, dp_inserts, dp_valid_inserts, d_num_insert,
            ycsb_config.num_txns * ycsb_config.num_ops_per_txn, thrust::identity<uint32_t>());

        logger.Trace("Allocating {} bytes for temp storage", formatSizeBytes(temp_storage_bytes));
        gpu_err_check(cudaMalloc(&d_temp_storage, temp_storage_bytes));

        logger.Info("Finished constructing TpccGpuIndex");
        size_t free, total;
        gpu_err_check(cudaMemGetInfo(&free, &total));
        logger.Info("GPU memory usage: {} / {}", formatSizeBytes(total - free), formatSizeBytes(total));
    }
    void loadInitialData()
    {
        auto &logger = Logger::GetInstance();
        logger.Info("Loading initial data");

        thrust::device_vector<uint32_t> d_keys(ycsb_config.starting_num_records);
        thrust::device_vector<uint32_t> d_values(ycsb_config.starting_num_records);
        thrust::sequence(d_keys.begin(), d_keys.end(), 0);
        thrust::sequence(d_values.begin(), d_values.end(), 0);
        auto zipped_kv = thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_values.begin()));
        index->insert(zipped_kv, zipped_kv + ycsb_config.starting_num_records);

        thrust::device_vector<uint32_t> found_values(ycsb_config.starting_num_records);
        index->find(d_keys.begin(), d_keys.end(), found_values.begin());
        if (thrust::equal(d_values.begin(), d_values.end(), found_values.begin())) {
            logger.Info("Initial data loaded successfully");
        } else {
            logger.Error("Initial data loaded incorrectly");
        }

        if (empty_key_sentinel != 0xffffffff) {
            logger.Error("empty_key_sentinel is not 0xffffffff");
        } else {
            logger.Info("empty_key_sentinel is 0xffffffff");
        }

        thrust::sequence(dp_free_rows, dp_free_rows + ycsb_config.num_records, ycsb_config.starting_num_records);
        logger.Info("Finished loading initial data");
        size_t free, total;
        gpu_err_check(cudaMemGetInfo(&free, &total));
        logger.Info("GPU memory usage: {} / {}", formatSizeBytes(total - free), formatSizeBytes(total));
    }

    void indexTxns(TxnArray<YcsbTxn> &txn_array, TxnArray<YcsbTxnParam> &index_array, uint32_t epoch_id)
    {
        if (txn_array.device != DeviceType::GPU || index_array.device != DeviceType::GPU)
        {
            throw std::runtime_error("TpccGpuIndex only supports GPU transaction array");
        }
        auto &logger = Logger::GetInstance();

        constexpr uint32_t block_size = 512;
        prepareYcsbIndexKernel<<<(ycsb_config.num_txns + block_size - 1) / block_size, block_size>>>(
            GpuTxnArray(txn_array), d_inserts, ycsb_config.num_txns);

        gpu_err_check(cudaPeekAtLastError());

        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_inserts, d_valid_inserts, d_num_insert,
            ycsb_config.num_txns * ycsb_config.num_ops_per_txn,
            [] __device__(uint32_t key) { return key != static_cast<uint32_t>(-1); });

        uint32_t num_inserts;
        gpu_err_check(cudaMemcpy(&num_inserts, d_num_insert, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        logger.Info("Found {} inserts", num_inserts);

        auto zipped_inserts =
            thrust::make_zip_iterator(thrust::make_tuple(dp_valid_inserts, dp_free_rows + free_start));
        index->insert(zipped_inserts, zipped_inserts + num_inserts);
        free_start += num_inserts;
        logger.Trace("Free rows used: {}", free_start);

        indexYcsbKernel<<<(ycsb_config.num_txns + block_size - 1) / block_size, block_size>>>(
            GpuTxnArray(txn_array), GpuTxnArray(index_array), index_view, ycsb_config.num_txns);
        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaDeviceSynchronize());
        logger.Info("Finished indexing transactions");
    }
};
} // namespace

YcsbGpuIndex::YcsbGpuIndex(YcsbConfig ycsb_config)
    : ycsb_config(ycsb_config)
{
    gpu_index_impl = std::make_any<YcsbGpuIndexImpl>(ycsb_config);
}
void YcsbGpuIndex::loadInitialData()
{
    auto &impl = std::any_cast<YcsbGpuIndexImpl &>(gpu_index_impl);
    impl.loadInitialData();
}
void YcsbGpuIndex::indexTxns(TxnArray<YcsbTxn> &txn_array, TxnArray<YcsbTxnParam> &index_array, uint32_t epoch_id)
{
    auto &impl = std::any_cast<YcsbGpuIndexImpl &>(gpu_index_impl);
    impl.indexTxns(txn_array, index_array, epoch_id);
}
} // namespace epic::ycsb