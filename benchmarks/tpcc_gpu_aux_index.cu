//
// Created by Shujian Qian on 2024-04-12.
//


#include <benchmarks/tpcc_gpu_aux_index.h>

#include <random>

#include <cooperative_groups.h>

#include <gpu_btree.h>
#include <benchmarks/tpcc_txn_gen.h>
#include <util_log.h>
#include <chrono>
#include <benchmarks/tpcc_txn.h>
#include <benchmarks/tpcc_gpu_txn.cuh>
#include <util_gpu_error_check.cuh>
#include <cub/warp/warp_merge_sort.cuh>

namespace epic::tpcc {

namespace detail {

namespace cg = cooperative_groups;

struct TreeParam
{
    using key_type = uint64_t;
    using value_type = uint64_t;
    using pair_type = var_pair_type<key_type, value_type, 42, 22>;
    constexpr static size_t branching_factor = 16;

    using node_type = GpuBTree::node_type<key_type, value_type, branching_factor, pair_type>;
    constexpr static size_t num_prealloc_nodes = 500000;
    using allocator_type = device_bump_allocator<node_type, num_prealloc_nodes>;
};

using CustomerOrderBTree = GpuBTree::gpu_blink_tree<TreeParam::key_type, TreeParam::value_type,
    TreeParam::branching_factor, TreeParam::allocator_type, TreeParam::pair_type>;

union PackedCustomerOrderKey
{
    using baseType = uint64_t;
    constexpr static baseType max_o_id = (1ull << 20) - 1;
    constexpr static baseType invalid_key = (1ull << 42) - 1;
    struct
    {
        baseType o_id : 20;
        baseType c_id : 12;
        baseType d_id : 4;
        baseType w_id : 6;
    };
    baseType base_key = 0;
    PackedCustomerOrderKey() = default;
    PackedCustomerOrderKey(baseType o_id, baseType c_id, baseType d_id, baseType w_id)
    {
        base_key = 0;
        this->o_id = o_id;
        this->c_id = c_id;
        this->d_id = d_id;
        this->w_id = w_id;
    }
};

template<typename T>
struct mapped_vector
{
    mapped_vector(std::size_t capacity)
        : capacity_(capacity)
    {
        allocate(capacity);
    }
    T &operator[](std::size_t index)
    {
        return dh_buffer_[index];
    }
    ~mapped_vector() {}
    void free()
    {
        cuda_try(cudaDeviceSynchronize());
        cuda_try(cudaFreeHost(dh_buffer_));
    }
    T *data() const
    {
        return dh_buffer_;
    }

    std::vector<T> to_std_vector()
    {
        std::vector<T> copy(capacity_);
        for (std::size_t i = 0; i < capacity_; i++)
        {
            copy[i] = dh_buffer_[i];
        }
        return copy;
    }

private:
    void allocate(std::size_t count)
    {
        cuda_try(cudaMallocHost(&dh_buffer_, sizeof(T) * count));
    }
    std::size_t capacity_;
    T *dh_buffer_;
};

template <typename GpuTxnArrayType>
void __global__ insert_txn_updates_kernel(GpuTxnArrayType txns, uint32_t num_txns, CustomerOrderBTree btree,
    CustomerOrderBTree::allocator_type host_allocator, uint32_t *order_num_items, uint32_t *order_customers,
    uint32_t (*order_items)[15], uint32_t num_slots_per_district)
{
    auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<CustomerOrderBTree::branching_factor>(block);

    if (thread_id - tile.thread_rank() >= num_txns)
    {
        return;
    }

    BaseTxn *base_txn_ptr = nullptr;
    uint32_t txn_type = 0xffffffffu;
    bool to_insert = false;
    if (thread_id < num_txns)
    {
        base_txn_ptr = txns.getTxn(thread_id);
        txn_type = base_txn_ptr->txn_type;
        to_insert = true;
    }
  using allocator_type = typename CustomerOrderBTree::device_allocator_context_type;
  allocator_type allocator{host_allocator, tile};

    uint32_t work_queue = tile.ballot(to_insert);
    while (work_queue)
    {
        int curr_rank = __ffs(work_queue) - 1;
        BaseTxn *curr_base_txn_ptr = tile.shfl(base_txn_ptr, curr_rank);
        uint32_t curr_txn_type = tile.shfl(txn_type, curr_rank);

        switch (static_cast<TpccTxnType>(curr_txn_type))
        {
        case TpccTxnType::NEW_ORDER: {
            auto no_txn_ptr = reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(curr_base_txn_ptr->data);
            PackedCustomerOrderKey key;
            key.w_id = no_txn_ptr->origin_w_id;
            key.d_id = no_txn_ptr->d_id;
            key.c_id = no_txn_ptr->c_id;
            key.o_id = PackedCustomerOrderKey::max_o_id - no_txn_ptr->o_id; // hack: to get backward scan

            uint32_t district_id = no_txn_ptr->d_id - 1 + (no_txn_ptr->origin_w_id - 1) * 10;
            uint32_t cache_idx = district_id * num_slots_per_district + no_txn_ptr->o_id;
            if (tile.thread_rank() == curr_rank)
            {
                order_num_items[cache_idx] = no_txn_ptr->num_items;
                order_customers[cache_idx] = no_txn_ptr->c_id;
            }
            if (tile.thread_rank() < no_txn_ptr->num_items)
            {
                order_items[cache_idx][tile.thread_rank()] = no_txn_ptr->items[tile.thread_rank()].i_id;
            }

            btree.cooperative_insert(key.base_key, 0ull, tile, allocator);
            break;
        }
        default:
            assert(false);
            break;
        }

        if (tile.thread_rank() == curr_rank) { to_insert = false; }
        work_queue = tile.ballot(to_insert);
    }
}

template <typename GpuTxnArrayType, typename GpuTxnIndexArrayType>
void __global__ perform_range_queries_kernel(GpuTxnArrayType txns, GpuTxnIndexArrayType index, uint32_t num_txns,
    CustomerOrderBTree btree, CustomerOrderBTree::allocator_type host_allocator, uint32_t *order_num_items,
    uint32_t *order_customers, uint32_t (*order_items)[15], uint32_t num_slots_per_district)
{
    auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<CustomerOrderBTree::branching_factor>(block);

    constexpr uint32_t warp_threads = 16;
    constexpr uint32_t block_threads = 512;
    constexpr uint32_t warps_per_block = block_threads / warp_threads;
    constexpr uint32_t items_per_thread = 19;
    const uint32_t warp_id = static_cast<uint32_t>(threadIdx.x) / warp_threads;
    using WarpMergeSortT = cub::WarpMergeSort<uint32_t, items_per_thread, warp_threads>;

    __shared__ typename WarpMergeSortT::TempStorage temp_storage[warps_per_block];

    if (thread_id - tile.thread_rank() >= num_txns)
    {
        return;
    }

    BaseTxn *base_txn_ptr = nullptr;
    BaseTxn *base_param_ptr = nullptr;
    uint32_t txn_type = 0xffffffffu;
    bool to_insert = false;
    if (thread_id < num_txns)
    {
        base_txn_ptr = txns.getTxn(thread_id);
        base_param_ptr = index.getTxn(thread_id);
        txn_type = base_txn_ptr->txn_type;
        to_insert = true;
    }
  using allocator_type = typename CustomerOrderBTree::device_allocator_context_type;
  allocator_type allocator{host_allocator, tile};

    uint32_t work_queue = tile.ballot(to_insert);
    while (work_queue)
    {
        int curr_rank = __ffs(work_queue) - 1;
        BaseTxn *curr_base_txn_ptr = tile.shfl(base_txn_ptr, curr_rank);
        BaseTxn *curr_base_param_ptr = tile.shfl(base_param_ptr, curr_rank);
        uint32_t curr_txn_type = tile.shfl(txn_type, curr_rank);

        switch (static_cast<TpccTxnType>(curr_txn_type))
        {
        case TpccTxnType::ORDER_STATUS: {
            auto os_txn_ptr = reinterpret_cast<OrderStatusTxnInput *>(curr_base_txn_ptr->data);
            PackedCustomerOrderKey lower_bound;
            lower_bound.w_id = os_txn_ptr->w_id;
            lower_bound.d_id = os_txn_ptr->d_id;
            lower_bound.c_id = os_txn_ptr->c_id;
            lower_bound.o_id = PackedCustomerOrderKey::max_o_id - os_txn_ptr->o_id;

            PackedCustomerOrderKey upper_bound;
            upper_bound.base_key = lower_bound.base_key;
            upper_bound.o_id = PackedCustomerOrderKey::max_o_id;

            CustomerOrderBTree::pair_type respair = btree.cooperative_find_next(lower_bound.base_key, upper_bound.base_key, tile, allocator);
            PackedCustomerOrderKey reskey;
            reskey.base_key = respair.first;

            uint32_t o_id = PackedCustomerOrderKey::max_o_id - reskey.o_id; // revert the hack

            /* verify against CPU aux index */
            // if (tile.thread_rank() == 0 && os_txn_ptr->o_id != o_id)
            // {
            //     printf("ERROR: txn[%d] cpu_oid[%d] gpu_oid[%d]\n", thread_id - tile.thread_rank() + curr_rank,
            //         os_txn_ptr->o_id, o_id);
            // }
            // assert(os_txn_ptr->o_id == o_id);

            uint32_t district_id = os_txn_ptr->d_id - 1 + (os_txn_ptr->w_id - 1) * 10;
            uint32_t cache_idx = district_id * num_slots_per_district + o_id;

            if (tile.thread_rank() == curr_rank)
            {
                os_txn_ptr->o_id = o_id;
                os_txn_ptr->num_items = order_num_items[cache_idx];
            }
            break;
        }
        case TpccTxnType::DELIVERY: {
            auto dl_txn_ptr = reinterpret_cast<DeliveryTxnInput *>(curr_base_txn_ptr->data);

            if (tile.thread_rank() < 10)
            {
                uint32_t district_id = tile.thread_rank() + (dl_txn_ptr->w_id - 1) * 10;
                uint32_t cache_idx = district_id * num_slots_per_district + dl_txn_ptr->o_id;

                /* verify against cpu aux index */
                // if (dl_txn_ptr->num_items[tile.thread_rank()] != order_num_items[cache_idx])
                // {
                //     printf("ERROR: delivery txn[%d] w[%d]d[%d] oid[%d] cpu_num_items[%d] gpu[%d]\n",
                //         thread_id - tile.thread_rank() + curr_rank, dl_txn_ptr->w_id, tile.thread_rank() + 1,
                //         dl_txn_ptr->o_id, dl_txn_ptr->num_items[tile.thread_rank()], order_num_items[cache_idx]);
                // }
                // if (dl_txn_ptr->customers[tile.thread_rank()] != order_customers[cache_idx])
                // {
                //     printf("ERROR: delivery txn[%d] w[%d]d[%d] cpu_customer[%d] gpu_[%d]\n",
                //         thread_id - tile.thread_rank() + curr_rank, dl_txn_ptr->w_id, tile.thread_rank() + 1,
                //         dl_txn_ptr->customers[tile.thread_rank()], order_customers[cache_idx]);
                // }

                dl_txn_ptr->num_items[tile.thread_rank()] = order_num_items[cache_idx];
                dl_txn_ptr->customers[tile.thread_rank()] = order_customers[cache_idx];
            }

            break;
        }
        case TpccTxnType::STOCK_LEVEL: {
            auto sl_txn_ptr = reinterpret_cast<StockLevelTxnInput *>(curr_base_txn_ptr->data);
            auto sl_param_ptr = reinterpret_cast<StockLevelTxnParams *>(curr_base_param_ptr->data);

            uint32_t items_cnt = 0;
            for (uint32_t o_id = sl_txn_ptr->o_id; o_id > sl_txn_ptr->o_id - 20; --o_id)
            {
                uint32_t district_id = sl_txn_ptr->d_id - 1 + (sl_txn_ptr->w_id - 1) * 10;
                uint32_t cache_idx = district_id * num_slots_per_district + o_id;

                uint32_t num_items = order_num_items[cache_idx];
                if (tile.thread_rank() < num_items)
                {
                    sl_param_ptr->stock_ids[items_cnt + tile.thread_rank()] = order_items[cache_idx][tile.thread_rank()];
                }
                items_cnt += num_items;
            }

            struct CustomLess
            {
                __device__ bool operator()(const uint32_t &lhs, const uint32_t &rhs)
                {
                    return lhs < rhs;
                }
            };
            WarpMergeSortT(temp_storage[warp_id])
                .Sort(
                    *reinterpret_cast<uint32_t(*)[19]>(&sl_param_ptr->stock_ids[items_per_thread * tile.thread_rank()]),
                    CustomLess(), items_cnt, 0xffffffffu);

            /* verify against CPU index */
            // if (tile.thread_rank() == 0)
            // {
            //     if (items_cnt != sl_txn_ptr->num_items)
            //     {
            //         printf("ERROR: stock_level txn[%d] w[%d] cpu_num_items[%d] gpu_[%d]\n",
            //             thread_id - tile.thread_rank() + curr_rank, sl_txn_ptr->w_id, sl_txn_ptr->num_items,
            //             items_cnt);
            //     }
            //     for (int i = 0; i < sl_txn_ptr->num_items; ++i)
            //     {
            //         if (sl_txn_ptr->o_id - 19 > 3000 && sl_txn_ptr->items[i] != sl_param_ptr->stock_ids[i])
            //         {
            //             printf("ERROR: stock_level txn[%d] w[%d]d[%d] item[%d] oid[%d] cpu_item[%d] gpu_[%d]\n",
            //                 thread_id - tile.thread_rank() + curr_rank, sl_txn_ptr->w_id, sl_txn_ptr->d_id, i,
            //                 sl_txn_ptr->o_id - 19, sl_txn_ptr->items[i], sl_param_ptr->stock_ids[i]);
            //         }
            //     }
            // }

            if (tile.thread_rank() == curr_rank)
            {
                sl_txn_ptr->num_items = items_cnt;
            }
            break;
        }
        default:
            assert(false);
            break;
        }

        if (tile.thread_rank() == curr_rank) { to_insert = false; }
        work_queue = tile.ballot(to_insert);
    }
}

class TpccGpuAuxIndexImpl
{
    TpccConfig config;
    CustomerOrderBTree co_btree;
    uint32_t num_slots_per_district;
    uint32_t num_slots;
    uint32_t *order_num_items, *order_customers;
    uint32_t (*order_items)[15];

public:
    explicit TpccGpuAuxIndexImpl(TpccConfig &config)
        : config{config}
        , num_slots_per_district(config.num_txns / config.num_warehouses / 10 * config.epochs + 3000)
        , num_slots(num_slots_per_district * config.num_warehouses * 10)
        , order_num_items(cuda_allocator<uint32_t>().allocate(num_slots))
        , order_customers(cuda_allocator<uint32_t>().allocate(num_slots))
        , order_items(cuda_allocator<uint32_t[15]>().allocate(num_slots))
    {}
    void loadInitialData()
    {
        mapped_vector<TreeParam::key_type> keys(config.num_warehouses * 10 * 3000);
        mapped_vector<TreeParam::value_type> values(config.num_warehouses * 10 * 3000);
        std::mt19937_64 gen(std::random_device{}());
        TpccNuRand i_id_dist(8191, 1, 100'000);

        std::vector<uint32_t> h_order_num_items(num_slots), h_order_customers(num_slots);
        uint32_t (*h_order_items)[15] = new uint32_t[num_slots][15];

        auto &logger = Logger::GetInstance();
        logger.Info("Loading gpu aux index");
        auto start = std::chrono::high_resolution_clock::now();
        uint32_t idx = 0;
        for (uint32_t w_id = 1; w_id <= config.num_warehouses; w_id++)
        {
            for (uint32_t d_id = 1; d_id <= 10; d_id++)
            {
                for (uint32_t o_id = 1; o_id <= 3000; o_id++)
                {
                    const uint32_t c_id = o_id; // assume one order per customer
                    const uint32_t sid = 0;     // initial sid
                    uint64_t hacked_oid = PackedCustomerOrderKey::max_o_id - o_id; // hack: to get backward scan
                    const PackedCustomerOrderKey key{hacked_oid, c_id, d_id, w_id};
                    keys[idx] = key.base_key;
                    values[idx] = sid;
                    ++idx;

                    const uint32_t o_num_items = 15;
                    const uint32_t district_id = d_id - 1+ (w_id - 1) * 10;
                    h_order_num_items[district_id * num_slots_per_district + o_id] = o_num_items; // TODO: randomize num_items for each order
                    h_order_customers[district_id * num_slots_per_district + o_id] = c_id;

                    uint32_t(&o_items)[15] = h_order_items[district_id * num_slots_per_district + o_id];
                    for (uint32_t ol_id = 0; ol_id < o_num_items; ++ol_id)
                    {
                        o_items[ol_id] = i_id_dist(gen);
                    }
                }
            }
        }
        co_btree.insert(keys.data(), values.data(), idx);
        gpu_err_check(cudaMemcpy(order_num_items, h_order_num_items.data(), sizeof(uint32_t) * num_slots, cudaMemcpyHostToDevice));
        gpu_err_check(cudaMemcpy(order_customers, h_order_customers.data(), sizeof(uint32_t) * num_slots, cudaMemcpyHostToDevice));
        gpu_err_check(cudaMemcpy(order_items, h_order_items, sizeof(uint32_t[15]) * num_slots, cudaMemcpyHostToDevice));

        delete []h_order_items;

        gpu_err_check(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        logger.Info(
            "Gpu aux index loaded in {}us", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        logger.Info("Validating gpu aux btree");
        std::vector<TreeParam::key_type> h_keys = keys.to_std_vector();
        // co_btree.validate_tree_structure(h_keys, [](const TreeParam::key_type &key) {return 0;});
        logger.Info("Validation passed: gpu aux btree");
    }

    template <typename TxnArrayType>
    void insertTxnUpdates(TxnArrayType &txns, size_t epoch)
    {
        uint32_t num_txns = txns.num_txns;
        const uint32_t block_size = 512;
        const uint32_t num_blocks = (num_txns + block_size - 1) / block_size;
        // TODO: revert hack
        insert_txn_updates_kernel<<<num_blocks, block_size>>>(GpuPackedTxnArray(txns), num_txns, co_btree,
            co_btree.get_allocator(), order_num_items, order_customers, order_items, num_slots_per_district);
        gpu_err_check(cudaDeviceSynchronize());
    }

    template <typename TxnArrayType, typename TxnParamArrayType>
    void performRangeQueries(TxnArrayType &txns, TxnParamArrayType &index, size_t epoch)
    {
        uint32_t num_txns = txns.num_txns;
        const uint32_t block_size = 512;
        const uint32_t num_blocks = (num_txns + block_size - 1) / block_size;
        // TODO: revert hack
        perform_range_queries_kernel<<<num_blocks, block_size>>>(GpuPackedTxnArray(txns), TpccGpuTxnArrayT(index), num_txns,
            co_btree, co_btree.get_allocator(), order_num_items, order_customers, order_items, num_slots_per_district);
        gpu_err_check(cudaDeviceSynchronize());
    }
};

} // namespace detail

template <typename TxnArrayType, typename TxnParamArrayType>
TpccGpuAuxIndex<TxnArrayType, TxnParamArrayType>::TpccGpuAuxIndex(TpccConfig &config)
    : impl{detail::TpccGpuAuxIndexImpl{config}}
{}

template <typename TxnArrayType, typename TxnParamArrayType>
void TpccGpuAuxIndex<TxnArrayType, TxnParamArrayType>::loadInitialData()
{
    std::any_cast<detail::TpccGpuAuxIndexImpl>(impl).loadInitialData();
}

template <typename TxnArrayType, typename TxnParamArrayType>
void TpccGpuAuxIndex<TxnArrayType, TxnParamArrayType>::insertTxnUpdates(TxnArrayType &txns, size_t epoch)
{
    std::any_cast<detail::TpccGpuAuxIndexImpl>(impl).insertTxnUpdates(txns, epoch);
}

template <typename TxnArrayType, typename TxnParamArrayType>
void TpccGpuAuxIndex<TxnArrayType, TxnParamArrayType>::performRangeQueries(
    TxnArrayType &txns, TxnParamArrayType &index, size_t epoch)
{
    std::any_cast<detail::TpccGpuAuxIndexImpl>(impl).performRangeQueries(txns, index, epoch);
}

/* instantiate the templated class with choosen type */
template class TpccGpuAuxIndex<TpccTxnArrayT, TpccTxnParamArrayT>;

} // namespace epic::tpcc
