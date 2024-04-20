//
// Created by Shujian Qian on 2024-04-20.
//

#ifndef MICRO_INDEX_H
#define MICRO_INDEX_H

#include <gpu_txn.cuh>
#include <util_gpu_error_check.cuh>
#include <cuco/static_map.cuh>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <benchmarks/micro_config.h>
#include <benchmarks/micro_txn.h>

namespace epic::micro {

using MicroIndexType = cuco::static_map<uint32_t, uint32_t>;
using MicroIndexDeviceView = MicroIndexType::device_view;

namespace detail {
static void __global__ indexMicroKernel(GpuTxnArray prev_txn, GpuTxnArray prev_param, bool *retry, GpuTxnArray curr_txn,
    GpuTxnArray curr_param, MicroIndexDeviceView index_view, uint32_t num_txns)
{

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns * 2)
    {
        return;
    }
    bool is_prev = tid < num_txns;
    /* if prev epoch's txn is not retried, then it is not indexed */
    if (is_prev && !retry[tid])
    {
        return;
    }

    uint32_t tid_in_epoch = is_prev ? tid : tid - num_txns;
    BaseTxn *base_txn_ptr = is_prev ? prev_txn.getTxn(tid_in_epoch) : curr_txn.getTxn(tid_in_epoch);
    BaseTxn *base_index_ptr = is_prev ? prev_param.getTxn(tid_in_epoch) : curr_param.getTxn(tid_in_epoch);
    MicroTxn *txn_ptr = reinterpret_cast<MicroTxn *>(base_txn_ptr->data);
    MicroTxnParams *index_ptr = reinterpret_cast<MicroTxnParams *>(base_index_ptr->data);

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
    }
    index_ptr->abort = txn_ptr->abort;
}
}

class MicroIndex
{
public:
    static constexpr cuco::empty_key<uint32_t> empty_key_sentinel{0xffffffff};
    static constexpr cuco::empty_value<uint32_t> empty_value_sentinel{0xffffffff};
    static constexpr double load_factor = 0.5;

    MicroConfig config;
    MicroIndexType index;
    MicroIndexDeviceView index_device_view;

    explicit MicroIndex(MicroConfig config)
        : config(config)
        , index(static_cast<size_t>(std::ceil(config.num_records / load_factor)), empty_key_sentinel,
              empty_value_sentinel)
        , index_device_view(index.get_device_view())
    {}

    void loadInitialData()
    {
        auto &logger = Logger::GetInstance();
        logger.Info("Loading initial data");

        thrust::device_vector<uint32_t> d_keys(config.num_records);
        thrust::device_vector<uint32_t> d_values(config.num_records);
        thrust::sequence(d_keys.begin(), d_keys.end(), 0);
        thrust::sequence(d_values.begin(), d_values.end(), 0);
        auto zipped_kv = thrust::make_zip_iterator(thrust::make_tuple(d_keys.begin(), d_values.begin()));
        index.insert(zipped_kv, zipped_kv + config.num_records);

        thrust::device_vector<uint32_t> found_values(config.num_records);
        index.find(d_keys.begin(), d_keys.end(), found_values.begin());
        if (thrust::equal(d_values.begin(), d_values.end(), found_values.begin()))
        {
            logger.Info("Initial data loaded successfully");
        }
        else
        {
            logger.Error("Initial data loaded incorrectly");
        }
    }

    void indexTxns(TxnArray<MicroTxn> &prev_txn, TxnArray<MicroTxnParams> &prev_param, bool *retry,
        TxnArray<MicroTxn> &curr_txn, TxnArray<MicroTxnParams> &curr_param)
    {
        auto &logger = Logger::GetInstance();
        constexpr uint32_t block_size = 512;
        const uint32_t num_blocks = (config.num_txns * 2 + block_size - 1) / block_size;

        detail::indexMicroKernel<<<num_blocks, block_size>>>(GpuTxnArray(prev_txn), GpuTxnArray(prev_param), retry,
            GpuTxnArray(curr_txn), GpuTxnArray(curr_param), index_device_view, config.num_txns);

        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaDeviceSynchronize());
        logger.Info("Finished indexing transactions");
    }
};

} // namespace epic::micro

#endif // MICRO_INDEX_H
