//
// Created by Shujian Qian on 2023-11-20.
//

#include <cmath>
#include <memory>
#include <cuda/std/atomic>

#include <thrust/device_vector.h>
#include <thrust/equal.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <benchmarks/tpcc_gpu_index.h>
#include <benchmarks/tpcc_table.h>
#include <gpu_txn.cuh>
#include <util_log.h>
#include <util_gpu_error_check.cuh>

#include <cuco/static_map.cuh>

#include <cub/cub.cuh>

namespace epic::tpcc {

namespace {
__device__ __forceinline__ void tpccPrepareInsertIndex(NewOrderTxnInput<FixedSizeTxn> *txn,
    OrderKey::baseType *order_insert, NewOrderKey::baseType *new_order_insert, OrderLineKey::baseType *orderline_insert,
    uint32_t tid)
{
    uint32_t num_items = txn->num_items;
    uint32_t o_id = txn->o_id;
    uint32_t d_id = txn->d_id;
    uint32_t origin_w_id = txn->origin_w_id;
    OrderKey order_key;
    order_key.o_id = o_id;
    order_key.o_d_id = d_id;
    order_key.o_w_id = origin_w_id;
    order_insert[tid] = order_key.base_key;
    NewOrderKey new_order_key;
    new_order_key.no_o_id = o_id;
    new_order_key.no_d_id = d_id;
    new_order_key.no_w_id = origin_w_id;
    new_order_insert[tid] = new_order_key.base_key;

    OrderLineKey order_line_key;
    order_line_key.ol_o_id = o_id;
    order_line_key.ol_d_id = d_id;
    uint32_t base = tid * 15;
    for (uint32_t i = 0; i < 15; ++i)
    {
        if (i >= num_items)
        {
            orderline_insert[base + i] = static_cast<OrderLineKey::baseType>(-1);
            continue;
        }
        order_line_key.ol_w_id = txn->items[i].w_id;
        order_line_key.ol_number = i + 1;
        orderline_insert[base + i] = order_line_key.base_key;
    }
}

__device__ __forceinline__ void tpccPrepareInsertIndex(PaymentTxnInput *txn, OrderKey::baseType *order_insert,
    NewOrderKey::baseType *new_order_insert, OrderLineKey::baseType *orderline_insert, uint32_t tid)
{

    order_insert[tid] = static_cast<OrderKey::baseType>(-1);
    new_order_insert[tid] = static_cast<NewOrderKey::baseType>(-1);
    for (uint32_t i = 0; i < 15; ++i)
    {
        orderline_insert[tid * 15 + i] = static_cast<OrderLineKey::baseType>(-1);
    }
}

__global__ void prepareTpccIndexKernel(GpuTxnArray txn, OrderKey::baseType *order_insert,
    NewOrderKey::baseType *new_order_insert, OrderLineKey::baseType *orderline_insert, uint32_t num_txns)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns)
    {
        return;
    }
    BaseTxn *txn_ptr = txn.getTxn(tid);

    switch (static_cast<TpccTxnType>(txn_ptr->txn_type))
    {
    case TpccTxnType::NEW_ORDER:
        tpccPrepareInsertIndex(reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(txn_ptr->data), order_insert,
            new_order_insert, orderline_insert, tid);
        break;
    case TpccTxnType::PAYMENT:
        tpccPrepareInsertIndex(
            reinterpret_cast<PaymentTxnInput *>(txn_ptr->data), order_insert, new_order_insert, orderline_insert, tid);
        break;
    default:
        /* TODO: implement prepare insert for the rest of txn types */
        break;
    }
}

using WarehouseIndexType = cuco::static_map<WarehouseKey::baseType, uint32_t>;
using DistrictIndexType = cuco::static_map<DistrictKey::baseType, uint32_t>;
using CustomerIndexType = cuco::static_map<CustomerKey::baseType, uint32_t>;
using HistoryIndexType = cuco::static_map<HistoryKey::baseType, uint32_t>;
using NewOrderIndexType = cuco::static_map<NewOrderKey::baseType, uint32_t>;
using OrderIndexType = cuco::static_map<OrderKey::baseType, uint32_t>;
using OrderLineIndexType = cuco::static_map<OrderLineKey::baseType, uint32_t>;
using ItemIndexType = cuco::static_map<ItemKey::baseType, uint32_t>;
using StockIndexType = cuco::static_map<StockKey::baseType, uint32_t>;

using WarehouseDeviceView = WarehouseIndexType::device_view;
using DistrictDeviceView = DistrictIndexType::device_view;
using CustomerDeviceView = CustomerIndexType::device_view;
using HistoryDeviceView = HistoryIndexType::device_view;
using NewOrderDeviceView = NewOrderIndexType::device_view;
using OrderDeviceView = OrderIndexType::device_view;
using OrderLineDeviceView = OrderLineIndexType::device_view;
using ItemDeviceView = ItemIndexType::device_view;
using StockDeviceView = StockIndexType::device_view;

struct tpccGpuIndexFindView
{
    WarehouseDeviceView warehouse_view;
    DistrictDeviceView district_view;
    CustomerDeviceView customer_view;
    HistoryDeviceView history_view;
    NewOrderDeviceView new_order_view;
    OrderDeviceView order_view;
    OrderLineDeviceView order_line_view;
    ItemDeviceView item_view;
    StockDeviceView stock_view;
};

void __device__ __forceinline__ indexTpccTxn(NewOrderTxnInput<FixedSizeTxn> *txn,
    NewOrderTxnParams<FixedSizeTxn> *index, tpccGpuIndexFindView index_view, uint32_t tid)
{
    {
        WarehouseKey warehouse_key;
        warehouse_key.key.w_id = txn->origin_w_id;
        auto warehouse_found = index_view.warehouse_view.find(warehouse_key.base_key);
        if (warehouse_found != index_view.warehouse_view.end())
        {
            index->warehouse_id = warehouse_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }

    {
        DistrictKey district_key;
        district_key.key.d_id = txn->d_id;
        district_key.key.d_w_id = txn->origin_w_id;
        auto district_found = index_view.district_view.find(district_key.base_key);
        if (district_found != index_view.district_view.end())
        {
            index->district_id = district_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }

    {
        CustomerKey customer_key;
        customer_key.key.c_id = txn->c_id;
        customer_key.key.c_d_id = txn->d_id;
        customer_key.key.c_w_id = txn->origin_w_id;
        auto customer_found = index_view.customer_view.find(customer_key.base_key);
        if (customer_found != index_view.customer_view.end())
        {
            index->customer_id = customer_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }

    {
        OrderKey order_key;
        order_key.o_id = txn->o_id;
        order_key.o_d_id = txn->d_id;
        order_key.o_w_id = txn->origin_w_id;

        auto order_found = index_view.order_view.find(order_key.base_key);
        if (order_found != index_view.order_view.end())
        {
            index->order_id = order_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }

    {
        NewOrderKey new_order_key;
        new_order_key.no_o_id = txn->o_id;
        new_order_key.no_d_id = txn->d_id;
        new_order_key.no_w_id = txn->origin_w_id;

        auto new_order_found = index_view.new_order_view.find(new_order_key.base_key);
        if (new_order_found != index_view.new_order_view.end())
        {
            index->new_order_id = new_order_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }

    {
        uint32_t num_items = txn->num_items;
        index->num_items = num_items;
        for (int i = 0; i < num_items; ++i)
        {
            {
                StockKey stock_key;
                stock_key.key.s_i_id = txn->items[i].i_id;
                stock_key.key.s_w_id = txn->items[i].w_id;
                auto stock_found = index_view.stock_view.find(stock_key.base_key);
                if (stock_found != index_view.stock_view.end())
                {
                    index->items[i].stock_id = stock_found->second.load(cuda::std::memory_order_relaxed);
                }
                else
                {
                    assert(false);
                }
            }
            {
                OrderLineKey order_line_key;
                order_line_key.ol_o_id = txn->o_id;
                order_line_key.ol_d_id = txn->d_id;
                order_line_key.ol_w_id = txn->items[i].w_id;
                order_line_key.ol_number = i + 1;
                auto order_line_found = index_view.order_line_view.find(order_line_key.base_key);
                if (order_line_found != index_view.order_line_view.end())
                {
                    index->items[i].order_line_id = order_line_found->second.load(cuda::std::memory_order_relaxed);
                }
                else
                {
                    assert(false);
                }
            }
            {
                ItemKey item_key;
                item_key.key.i_id = txn->items[i].i_id;
                auto item_found = index_view.item_view.find(item_key.base_key);
                if (item_found != index_view.item_view.end())
                {
                    index->items[i].item_id = item_found->second.load(cuda::std::memory_order_relaxed);
                }
                else
                {
                    assert(false);
                }
            }
        }
    }
}

void __device__ __forceinline__ indexTpccTxn(
    PaymentTxnInput *txn, PaymentTxnParams *index, tpccGpuIndexFindView index_view, uint32_t tid)
{
    {
        WarehouseKey warehouse_key;
        warehouse_key.key.w_id = txn->warehouse_id;
        auto warehouse_found = index_view.warehouse_view.find(warehouse_key.base_key);
        if (warehouse_found != index_view.warehouse_view.end())
        {
            index->warehouse_id = warehouse_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }

    {
        DistrictKey district_key;
        district_key.key.d_id = txn->district_id;
        district_key.key.d_w_id = txn->warehouse_id;
        auto district_found = index_view.district_view.find(district_key.base_key);
        if (district_found != index_view.district_view.end())
        {
            index->district_id = district_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }

    {
        CustomerKey customer_key;
        customer_key.key.c_id = txn->customer_id;
        customer_key.key.c_d_id = txn->customer_district_id;
        customer_key.key.c_w_id = txn->customer_warehouse_id;
        auto customer_found = index_view.customer_view.find(customer_key.base_key);
        if (customer_found != index_view.customer_view.end())
        {
            index->customer_id = customer_found->second.load(cuda::std::memory_order_relaxed);
        }
        else
        {
            assert(false);
        }
    }
}

__global__ void indexTpccTxnKernel(
    GpuTxnArray txn, GpuTxnArray index, tpccGpuIndexFindView index_view, uint32_t num_txns)
{

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_txns)
    {
        return;
    }
    BaseTxn *txn_ptr = txn.getTxn(tid);
    BaseTxn *index_ptr = index.getTxn(tid);

    index_ptr->txn_type = txn_ptr->txn_type;
    switch (static_cast<TpccTxnType>(txn_ptr->txn_type))
    {
    case TpccTxnType::NEW_ORDER:
        indexTpccTxn(reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(txn_ptr->data),
            reinterpret_cast<NewOrderTxnParams<FixedSizeTxn> *>(index_ptr->data), index_view, tid);
        break;
    case TpccTxnType::PAYMENT:
        indexTpccTxn(reinterpret_cast<PaymentTxnInput *>(txn_ptr->data),
            reinterpret_cast<PaymentTxnParams *>(index_ptr->data), index_view, tid);
        break;
    default:
        /* TODO: implement prepare insert for the rest of txn types */
        break;
    }
}

template<typename InputType>
class DummyPredicate
{
public:
    __device__ __forceinline__ bool operator()(InputType val)
    {
        return true;
    }
};
} // namespace

class TpccGpuIndexImpl
{
public:
    static constexpr double load_factor = 0.5;

    static constexpr cuco::empty_key<WarehouseKey::baseType> warehouse_key_sentinel{
        static_cast<WarehouseKey::baseType>(-1)};
    static constexpr cuco::empty_key<DistrictKey::baseType> district_key_sentinel{
        static_cast<DistrictKey::baseType>(-1)};
    static constexpr cuco::empty_key<CustomerKey::baseType> customer_key_sentinel{
        static_cast<CustomerKey::baseType>(-1)};
    static constexpr cuco::empty_key<HistoryKey::baseType> history_key_sentinel{static_cast<HistoryKey::baseType>(-1)};
    static constexpr cuco::empty_key<NewOrderKey::baseType> new_order_key_sentinel{
        static_cast<NewOrderKey::baseType>(-1)};
    static constexpr cuco::empty_key<OrderKey::baseType> order_key_sentinel{static_cast<OrderKey::baseType>(-1)};
    static constexpr cuco::empty_key<OrderLineKey::baseType> order_line_key_sentinel{
        static_cast<OrderLineKey::baseType>(-1)};
    static constexpr cuco::empty_key<ItemKey::baseType> item_key_sentinel{static_cast<ItemKey::baseType>(-1)};
    static constexpr cuco::empty_key<StockKey::baseType> stock_key_sentinel{static_cast<StockKey::baseType>(-1)};

    static constexpr cuco::empty_value<uint32_t> value_sentinel{static_cast<uint32_t>(-1)};

    TpccConfig tpcc_config;
    std::shared_ptr<WarehouseIndexType> warehouse_index;
    std::shared_ptr<DistrictIndexType> district_index;
    std::shared_ptr<CustomerIndexType> customer_index;
    std::shared_ptr<HistoryIndexType> history_index;
    std::shared_ptr<NewOrderIndexType> new_order_index;
    std::shared_ptr<OrderIndexType> order_index;
    std::shared_ptr<OrderLineIndexType> order_line_index;
    std::shared_ptr<ItemIndexType> item_index;
    std::shared_ptr<StockIndexType> stock_index;
    tpccGpuIndexFindView index_device_view; /* the order is crucial so that it's initialized after the indices */

    uint32_t *d_order_free_rows;
    uint32_t *d_new_order_free_rows;
    uint32_t *d_order_line_free_rows;
    thrust::device_ptr<uint32_t> dp_order_free_rows;
    thrust::device_ptr<uint32_t> dp_new_order_free_rows;
    thrust::device_ptr<uint32_t> dp_order_line_free_rows;
    uint32_t order_free_start = 0;
    uint32_t new_order_free_start = 0;
    uint32_t order_line_free_start = 0;

    OrderKey::baseType *d_order_insert, *d_order_valid_insert;
    NewOrderKey::baseType *d_new_order_insert, *d_new_order_valid_insert;
    OrderLineKey::baseType *d_order_line_insert, *d_order_line_valid_insert;
    thrust::device_ptr<OrderKey::baseType> dp_order_valid_insert;
    thrust::device_ptr<NewOrderKey::baseType> dp_new_order_valid_insert;
    thrust::device_ptr<OrderLineKey::baseType> dp_order_line_valid_insert;
    uint32_t *d_order_num_insert, *d_new_order_num_insert, *d_order_line_num_insert;

    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    explicit TpccGpuIndexImpl(TpccConfig tpcc_config)
        : tpcc_config(tpcc_config)
        , warehouse_index{std::make_shared<WarehouseIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.warehouseTableSize() / load_factor)), warehouse_key_sentinel,
              value_sentinel)}
        , district_index{std::make_shared<DistrictIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.districtTableSize() / load_factor)), district_key_sentinel,
              value_sentinel)}
        , customer_index{std::make_shared<CustomerIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.customerTableSize() / load_factor)), customer_key_sentinel,
              value_sentinel)}
        , history_index{std::make_shared<HistoryIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.historyTableSize() / load_factor)), history_key_sentinel,
              value_sentinel)}
        , new_order_index{std::make_shared<NewOrderIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.newOrderTableSize() / load_factor)), new_order_key_sentinel,
              value_sentinel)}
        , order_index{std::make_shared<OrderIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.orderTableSize() / load_factor)), order_key_sentinel,
              value_sentinel)}
        , order_line_index{std::make_shared<OrderLineIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.orderLineTableSize() / load_factor)), order_line_key_sentinel,
              value_sentinel)}
        , item_index{std::make_shared<ItemIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.itemTableSize() / load_factor)), item_key_sentinel,
              value_sentinel)}
        , stock_index{std::make_shared<StockIndexType>(
              static_cast<size_t>(std::ceil(tpcc_config.stockTableSize() / load_factor)), stock_key_sentinel,
              value_sentinel)}
        , index_device_view{warehouse_index->get_device_view(), district_index->get_device_view(),
              customer_index->get_device_view(), history_index->get_device_view(), new_order_index->get_device_view(),
              order_index->get_device_view(), order_line_index->get_device_view(), item_index->get_device_view(),
              stock_index->get_device_view()}
    {
        auto &logger = Logger::GetInstance();

        gpu_err_check(cudaMalloc(&d_order_free_rows, tpcc_config.orderTableSize() * sizeof(uint32_t)));
        dp_order_free_rows = thrust::device_pointer_cast(d_order_free_rows);
        gpu_err_check(cudaMalloc(&d_new_order_free_rows, tpcc_config.newOrderTableSize() * sizeof(uint32_t)));
        dp_new_order_free_rows = thrust::device_pointer_cast(d_new_order_free_rows);
        gpu_err_check(cudaMalloc(&d_order_line_free_rows, tpcc_config.orderLineTableSize() * sizeof(uint32_t)));
        dp_order_line_free_rows = thrust::device_pointer_cast(d_order_line_free_rows);

        gpu_err_check(cudaMalloc(&d_order_insert, tpcc_config.num_txns * sizeof(OrderKey::baseType)));
        gpu_err_check(cudaMalloc(&d_order_valid_insert, tpcc_config.num_txns * sizeof(OrderKey::baseType)));
        dp_order_valid_insert = thrust::device_pointer_cast(d_order_valid_insert);
        gpu_err_check(cudaMalloc(&d_new_order_insert, tpcc_config.num_txns * sizeof(NewOrderKey::baseType)));
        gpu_err_check(cudaMalloc(&d_new_order_valid_insert, tpcc_config.num_txns * sizeof(NewOrderKey::baseType)));
        dp_new_order_valid_insert = thrust::device_pointer_cast(d_new_order_valid_insert);
        gpu_err_check(cudaMalloc(&d_order_line_insert, tpcc_config.num_txns * 15 * sizeof(OrderLineKey::baseType)));
        gpu_err_check(
            cudaMalloc(&d_order_line_valid_insert, tpcc_config.num_txns * 15 * sizeof(OrderLineKey::baseType)));
        dp_order_line_valid_insert = thrust::device_pointer_cast(d_order_line_valid_insert);
        gpu_err_check(cudaMalloc(&d_order_num_insert, sizeof(uint32_t)));
        gpu_err_check(cudaMalloc(&d_new_order_num_insert, sizeof(uint32_t)));
        gpu_err_check(cudaMalloc(&d_order_line_num_insert, sizeof(uint32_t)));

        size_t max_bytes = 0;
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_order_insert, d_order_valid_insert,
            d_order_num_insert, tpcc_config.num_txns, DummyPredicate<OrderKey::baseType>());
        max_bytes = std::max(max_bytes, temp_storage_bytes);

        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_new_order_insert, d_new_order_valid_insert,
            d_new_order_num_insert, tpcc_config.num_txns, DummyPredicate<NewOrderKey::baseType>());
        max_bytes = std::max(max_bytes, temp_storage_bytes);

        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_order_line_insert, d_order_line_valid_insert,
            d_order_line_num_insert, tpcc_config.num_txns * 15, DummyPredicate<OrderLineKey::baseType>());
        max_bytes = std::max(max_bytes, temp_storage_bytes);

        temp_storage_bytes = max_bytes;
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

        {

            logger.Trace("Loading warehouse table");
            std::vector<WarehouseKey::baseType> warehouse_keys;
            warehouse_keys.reserve(tpcc_config.num_warehouses);
            for (uint32_t i = 0; i < tpcc_config.num_warehouses; ++i)
            {
                warehouse_keys.push_back(WarehouseKey{i + 1}.base_key);
            }
            thrust::device_vector<WarehouseKey::baseType> d_warehouse_keys(warehouse_keys);
            thrust::device_vector<uint32_t> d_warehouse_values(tpcc_config.num_warehouses);
            thrust::sequence(d_warehouse_values.begin(), d_warehouse_values.end(), 0);
            auto zipped_warehouse_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_warehouse_keys.begin(), d_warehouse_values.begin()));
            warehouse_index->insert(zipped_warehouse_kv, zipped_warehouse_kv + tpcc_config.num_warehouses);
        }

        {
            logger.Trace("Loading district table");
            size_t num_districts = tpcc_config.num_warehouses * 10;
            std::vector<DistrictKey::baseType> district_keys;
            district_keys.reserve(num_districts);
            for (uint32_t w_id = 1; w_id <= tpcc_config.num_warehouses; ++w_id)
            {
                for (uint32_t d_id = 1; d_id <= 10; ++d_id)
                {
                    district_keys.push_back(DistrictKey{d_id, w_id}.base_key);
                }
            }
            thrust::device_vector<DistrictKey::baseType> d_district_keys(district_keys);
            thrust::device_vector<uint32_t> d_district_values(num_districts);
            thrust::sequence(d_district_values.begin(), d_district_values.end(), 0);
            auto zipped_district_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_district_keys.begin(), d_district_values.begin()));
            district_index->insert(zipped_district_kv, zipped_district_kv + num_districts);
        }

        {
            logger.Trace("Loading customer table");
            size_t num_customers = tpcc_config.num_warehouses * 10 * 3000;
            std::vector<CustomerKey::baseType> customer_keys;
            customer_keys.reserve(num_customers);
            for (uint32_t w_id = 1; w_id <= tpcc_config.num_warehouses; ++w_id)
            {
                for (uint32_t d_id = 1; d_id <= 10; ++d_id)
                {
                    for (uint32_t c_id = 1; c_id <= 3000; ++c_id)
                    {
                        customer_keys.push_back(CustomerKey{c_id, d_id, w_id}.base_key);
                    }
                }
            }
            thrust::device_vector<CustomerKey::baseType> d_customer_keys(customer_keys);
            thrust::device_vector<uint32_t> d_customer_values(num_customers);
            thrust::sequence(d_customer_values.begin(), d_customer_values.end(), 0);
            auto zipped_customer_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_customer_keys.begin(), d_customer_values.begin()));
            customer_index->insert(zipped_customer_kv, zipped_customer_kv + num_customers);
        }

        {
            /* TODO: populate data in History Table */
        }

        {
            logger.Trace("Loading item table");
            size_t num_items = 100'000;
            std::vector<ItemKey::baseType> item_keys;
            item_keys.reserve(num_items);
            for (uint32_t i_id = 1; i_id <= 100'000; ++i_id)
            {
                item_keys.push_back(ItemKey{i_id}.base_key);
            }
            thrust::device_vector<ItemKey::baseType> d_item_keys(item_keys);
            thrust::device_vector<uint32_t> d_item_values(num_items);
            thrust::sequence(d_item_values.begin(), d_item_values.end(), 0);
            auto zipped_item_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_item_keys.begin(), d_item_values.begin()));
            item_index->insert(zipped_item_kv, zipped_item_kv + num_items);
        }

        {
            logger.Trace("Loading stock table");
            size_t num_stocks = tpcc_config.num_warehouses * 100'000;
            std::vector<StockKey::baseType> stock_keys;
            stock_keys.reserve(num_stocks);
            for (uint32_t w_id = 1; w_id <= tpcc_config.num_warehouses; ++w_id)
            {
                for (uint32_t i_id = 1; i_id <= 100'000; ++i_id)
                {
                    stock_keys.push_back(StockKey{i_id, w_id}.base_key);
                }
            }
            thrust::device_vector<StockKey::baseType> d_stock_keys(stock_keys);
            thrust::device_vector<uint32_t> d_stock_values(num_stocks);
            thrust::sequence(d_stock_values.begin(), d_stock_values.end(), 0);
            auto zipped_stock_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_stock_keys.begin(), d_stock_values.begin()));
            stock_index->insert(zipped_stock_kv, zipped_stock_kv + num_stocks);
        }

        {
            logger.Trace("Loading order table");
            size_t num_orders = tpcc_config.num_warehouses * 10 * 3'000;
            size_t num_new_orders = tpcc_config.num_warehouses * 10 * 900;
            size_t num_order_lines = tpcc_config.num_warehouses * 10 * 3'000 * 15;
            std::vector<OrderKey::baseType> order_keys;
            std::vector<NewOrderKey::baseType> new_order_keys;
            std::vector<OrderLineKey::baseType> order_line_keys;
            order_keys.reserve(num_orders);
            new_order_keys.reserve(num_new_orders);
            order_line_keys.reserve(num_order_lines);
            for (uint32_t w_id = 1; w_id <= tpcc_config.num_warehouses; ++w_id)
            {
                for (uint32_t d_id = 1; d_id <= 10; ++d_id)
                {
                    for (uint32_t o_id = 1; o_id <= 3'000; ++o_id)
                    {
                        order_keys.push_back(OrderKey{o_id, d_id, w_id}.base_key);
                        if (o_id > 2'100)
                        {
                            new_order_keys.push_back(NewOrderKey{o_id, d_id, w_id}.base_key);
                        }
                        for (uint32_t ol_number = 1; ol_number <= 15; ++ol_number)
                        {
                            order_line_keys.push_back(OrderLineKey{o_id, d_id, w_id, ol_number}.base_key);
                        }
                    }
                }
            }

            thrust::device_vector<OrderKey::baseType> d_order_keys(order_keys);
            thrust::device_vector<NewOrderKey::baseType> d_new_order_keys(new_order_keys);
            thrust::device_vector<OrderLineKey::baseType> d_order_line_keys(order_line_keys);

            thrust::device_vector<uint32_t> d_order_values(num_orders);
            thrust::device_vector<uint32_t> d_new_order_values(num_new_orders);
            thrust::device_vector<uint32_t> d_order_line_values(num_order_lines);

            thrust::sequence(d_order_values.begin(), d_order_values.end(), 0);
            thrust::sequence(d_new_order_values.begin(), d_new_order_values.end(), 0);
            thrust::sequence(d_order_line_values.begin(), d_order_line_values.end(), 0);

            auto zipped_order_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_order_keys.begin(), d_order_values.begin()));
            auto zipped_new_order_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_new_order_keys.begin(), d_new_order_values.begin()));
            auto zipped_order_line_kv =
                thrust::make_zip_iterator(thrust::make_tuple(d_order_line_keys.begin(), d_order_line_values.begin()));

            order_index->insert(zipped_order_kv, zipped_order_kv + num_orders);
            new_order_index->insert(zipped_new_order_kv, zipped_new_order_kv + num_new_orders);
            order_line_index->insert(zipped_order_line_kv, zipped_order_line_kv + num_order_lines);

            thrust::sequence(dp_order_free_rows, dp_order_free_rows + tpcc_config.orderTableSize(), num_orders);
            thrust::sequence(
                dp_new_order_free_rows, dp_new_order_free_rows + tpcc_config.newOrderTableSize(), num_new_orders);
            thrust::sequence(
                dp_order_line_free_rows, dp_order_line_free_rows + tpcc_config.orderLineTableSize(), num_order_lines);
        }

        logger.Info("Finished loading initial data");
        size_t free, total;
        gpu_err_check(cudaMemGetInfo(&free, &total));
        logger.Info("GPU memory usage: {} / {}", formatSizeBytes(total - free), formatSizeBytes(total));

#if 0 // DEBUG
        {
            /* test inserted KV's */
            std::vector<StockKey::baseType> stock_keys;
            for (uint32_t i = 0; i < tpcc_config.num_warehouses; ++i)
            {
                for (uint32_t j = 0; j < 100'000; j += 1000)
                {
                    stock_keys.push_back(StockKey{j + 1, i + 1}.base_key);
                }
            }
            size_t num_stocks = stock_keys.size();
            thrust::device_vector<StockKey::baseType> d_stock_keys(stock_keys);
            thrust::device_vector<uint32_t> d_stock_values(num_stocks);
            std::vector<uint32_t> h_stock_values(num_stocks);
            stock_index->find(d_stock_keys.begin(), d_stock_keys.end(), d_stock_values.begin());
            thrust::copy(d_stock_values.begin(), d_stock_values.end(), h_stock_values.begin());
            auto &logger = Logger::GetInstance();
            for (size_t i = 0; i < num_stocks; ++i)
            {
                StockKey stock_key{0, 0};
                stock_key.base_key = stock_keys[i];
                logger.Info("Stock key: w_id[{}] i_id[{}], value: {}", stock_key.key.s_w_id, stock_key.key.s_i_id,
                    h_stock_values[i]);
            }
        }
        {
            std::vector<OrderKey::baseType> order_keys;
                for (uint32_t i = 0; i < tpcc_config.num_warehouses; ++i)
                {
                for (uint32_t j = 0; j < 10; ++j)
                {
                    for (uint32_t k = 0; k < 3'000; k += 1000)
                    {
                        order_keys.push_back(OrderKey{k + 1, j + 1, i + 1}.base_key);
                    }
                }
                }
                size_t num_orders = order_keys.size();
                thrust::device_vector<OrderKey::baseType> d_order_keys(order_keys);
                thrust::device_vector<uint32_t> d_order_values(num_orders);
                std::vector<uint32_t> h_order_values(num_orders);
                order_index->find(d_order_keys.begin(), d_order_keys.end(), d_order_values.begin());
                thrust::copy(d_order_values.begin(), d_order_values.end(), h_order_values.begin());
                auto &logger = Logger::GetInstance();
                for (size_t i = 0; i < num_orders; ++i)
                {
                    OrderKey order_key{0, 0, 0};
                    order_key.base_key = order_keys[i];
                    logger.Info("Order key: w_id[{}] d_id[{}] o_id[{}], value: {}", order_key.o_w_id,
                        order_key.o_d_id, order_key.o_id, h_order_values[i]);
                }
        }
#endif
    }

    void indexTxns(TxnArray<TpccTxn> &txn_array, TxnArray<TpccTxnParam> &index_array, uint32_t epoch_id)
    {
        if (txn_array.device != DeviceType::GPU || index_array.device != DeviceType::GPU)
        {
            throw std::runtime_error("TpccGpuIndex only supports GPU transaction array");
        }
        auto &logger = Logger::GetInstance();

        constexpr uint32_t block_size = 512;
        prepareTpccIndexKernel<<<(tpcc_config.num_txns + block_size - 1) / block_size, block_size>>>(
            GpuTxnArray(txn_array), d_order_insert, d_new_order_insert, d_order_line_insert, tpcc_config.num_txns);

        gpu_err_check(cudaPeekAtLastError());

        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_order_insert, d_order_valid_insert,
            d_order_num_insert, tpcc_config.num_txns,
            [] __device__(OrderKey::baseType val) { return val != static_cast<OrderKey::baseType>(-1); });
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_new_order_insert, d_new_order_valid_insert,
            d_new_order_num_insert, tpcc_config.num_txns,
            [] __device__(NewOrderKey::baseType val) { return val != static_cast<NewOrderKey::baseType>(-1); });
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_order_line_insert, d_order_line_valid_insert,
            d_order_line_num_insert, tpcc_config.num_txns * 15,
            [] __device__(OrderLineKey::baseType val) { return val != static_cast<OrderLineKey::baseType>(-1); });

        uint32_t num_orders_inserts, num_new_orders_inserts, num_order_lines_inserts;
        gpu_err_check(cudaMemcpy(&num_orders_inserts, d_order_num_insert, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpu_err_check(
            cudaMemcpy(&num_new_orders_inserts, d_new_order_num_insert, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        gpu_err_check(
            cudaMemcpy(&num_order_lines_inserts, d_order_line_num_insert, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        logger.Trace("Number of orders inserts: {}", num_orders_inserts);
        logger.Trace("Number of new orders inserts: {}", num_new_orders_inserts);
        logger.Trace("Number of order lines inserts: {}", num_order_lines_inserts);

        auto zipped_order_insert =
            thrust::make_zip_iterator(thrust::make_tuple(dp_order_valid_insert, dp_order_free_rows + order_free_start));
        auto zipped_new_order_insert = thrust::make_zip_iterator(
            thrust::make_tuple(dp_new_order_valid_insert, dp_new_order_free_rows + new_order_free_start));
        auto zipped_order_line_insert = thrust::make_zip_iterator(
            thrust::make_tuple(dp_order_line_valid_insert, dp_order_line_free_rows + order_line_free_start));
        order_index->insert(zipped_order_insert, zipped_order_insert + num_orders_inserts);
        new_order_index->insert(zipped_new_order_insert, zipped_new_order_insert + num_new_orders_inserts);
        order_line_index->insert(zipped_order_line_insert, zipped_order_line_insert + num_order_lines_inserts);
        order_free_start += num_orders_inserts;
        new_order_free_start += num_new_orders_inserts;
        order_line_free_start += num_order_lines_inserts;
        logger.Trace("Order free rows used: {}", order_free_start);
        logger.Trace("New order free rows used: {}", new_order_free_start);
        logger.Trace("Order line free rows used: {}", order_line_free_start);

        indexTpccTxnKernel<<<(tpcc_config.num_txns + block_size - 1) / block_size, block_size>>>(
            GpuTxnArray(txn_array), GpuTxnArray(index_array), index_device_view, tpcc_config.num_txns);
        gpu_err_check(cudaPeekAtLastError());
        gpu_err_check(cudaDeviceSynchronize());
        logger.Info("Finished indexing transactions");
    }
};

TpccGpuIndex::TpccGpuIndex(TpccConfig tpcc_config)
    : tpcc_config(tpcc_config)
{
    gpu_index_impl = std::make_any<TpccGpuIndexImpl>(tpcc_config);
}

void TpccGpuIndex::loadInitialData()
{
    auto &impl = std::any_cast<TpccGpuIndexImpl &>(gpu_index_impl);
    impl.loadInitialData();
}

void TpccGpuIndex::indexTxns(TxnArray<TpccTxn> &txn_array, TxnArray<TpccTxnParam> &index_array, uint32_t epoch_id)
{
    auto &impl = std::any_cast<TpccGpuIndexImpl &>(gpu_index_impl);
    impl.indexTxns(txn_array, index_array, epoch_id);
}

} // namespace epic::tpcc