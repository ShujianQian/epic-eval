//
// Created by Shujian Qian on 2023-10-25.
//

#include <benchmarks/tpcc_gpu_executor.h>

#include <stdio.h>

#include <gpu_storage.cuh>
#include <util_gpu_error_check.cuh>
#include <util_arch.h>
#include <gpu_txn.cuh>
#include <util_warp_memory.cuh>
#include <util_gpu_transfer.h>
#include <util_log.h>

namespace epic::tpcc {

namespace {

constexpr uint32_t block_size = 128;
static_assert(block_size % kDeviceWarpSize == 0, "block_size must be a multiple of 32");
constexpr uint32_t num_warps = block_size / kDeviceWarpSize;

__device__ uint32_t txn_counter = 0; /* used for scheduling txns among threads */
const uint32_t zero = 0;

__device__ __forceinline__ void gpuExecTpccTxn(TpccRecords records, TpccVersions versions,
    NewOrderTxnParams<FixedSizeTxn> *params, NewOrderExecPlan<FixedSizeTxn> *plan, uint32_t epoch, uint32_t lane_id,
    uint32_t txn_id /* for debug TODO: remove*/)
{
    constexpr uint32_t leader_lane = 0;
    constexpr uint32_t all_lanes_mask = 0xffffffffu;
    constexpr uint32_t s_quantity_offset = offsetof(StockValue, s_quantity) / sizeof(uint32_t);
    constexpr uint32_t d_next_o_id_offset = offsetof(DistrictValue, d_next_o_id) / sizeof(uint32_t);

#if 0 // DEBUG
    if (lane_id == leader_lane)
//        if (txn_id < 100 && lane_id == leader_lane)
    {
        printf("neworder txn_id[%u] warehouse i[%u]l[%u] district i[%u]l[%u] "
               "num_items[%u] "
               "item1[%u] stock_read1[%u] stock_write1[%u] orderline1_write[%u]"
               "\n",
            txn_id, params->warehouse_id, plan->warehouse_loc, params->district_id, plan->district_loc,
            params->num_items, plan->item_plans[0].item_loc, plan->item_plans[0].stock_read_loc,
            plan->item_plans[0].stock_write_loc, plan->item_plans[0].orderline_loc);
    }
#endif

    uint32_t result = 0;
    gpuReadFromTableCoop(records.warehouse_record, versions.warehouse_version, params->warehouse_id,
        plan->warehouse_loc, epoch, result, lane_id);

    gpuReadFromTableCoop(records.district_record, versions.district_version, params->district_id, plan->district_loc,
        epoch, result, lane_id);
    if (lane_id == d_next_o_id_offset)
    {
        // printf("RESULT[%u] tid[%u] district[%u] next_o_id[%u]\n", result, txn_id, params->district_id, params->next_order_id);
        result = params->next_order_id;
    }
    gpuWriteToTableCoop(records.district_record, versions.district_version, params->district_id,
        plan->district_write_loc, epoch, result, lane_id);

    gpuReadFromTableCoop(records.customer_record, versions.customer_version, params->customer_id, plan->customer_loc,
        epoch, result, lane_id);

    gpuWriteToTableCoop(
        records.order_record, versions.order_version, params->order_id, plan->order_loc, epoch, result, lane_id);

    gpuWriteToTableCoop(records.new_order_record, versions.new_order_version, params->new_order_id, plan->new_order_loc,
        epoch, result, lane_id);

    for (uint32_t i = 0; i < params->num_items; ++i)
    {
        gpuReadFromTableCoop(records.item_record, versions.item_version, params->items[i].item_id,
            plan->item_plans[i].item_loc, epoch, result, lane_id);
        gpuReadFromTableCoop(records.stock_record, versions.stock_version, params->items[i].stock_id,
            plan->item_plans[i].stock_read_loc, epoch, result, lane_id);
        if (lane_id == s_quantity_offset)
        {
            uint32_t order_quantity = params->items[i].order_quantities;
            result = result > order_quantity + 10 ? result - order_quantity : result + 91 - order_quantity;
        }
        gpuWriteToTableCoop(records.stock_record, versions.stock_version, params->items[i].stock_id,
            plan->item_plans[i].stock_write_loc, epoch, result, lane_id);

        constexpr uint32_t ol_i_id_offset = offsetof(OrderLineValue, ol_i_id) / sizeof(uint32_t);
        constexpr uint32_t ol_amount_offset = offsetof(OrderLineValue, ol_amount) / sizeof(uint32_t);
        constexpr uint32_t ol_supply_w_id_offset = offsetof(OrderLineValue, ol_supply_w_id) / sizeof(uint32_t);
        constexpr uint32_t ol_quantity_offset = offsetof(OrderLineValue, ol_quantity) / sizeof(uint32_t);
        if (lane_id == ol_i_id_offset)
        {
            result = params->items[i].item_id;
        }
        if (lane_id == ol_amount_offset)
        {
            result = params->items[i].order_quantities;
        }
        if (lane_id == ol_supply_w_id_offset)
        {
            result = params->warehouse_id;
        }
        if (lane_id == ol_quantity_offset)
        {
            result = params->items[i].order_quantities;
        }
        gpuWriteToTableCoop(records.order_line_record, versions.order_line_version, params->items[i].order_line_id,
            plan->item_plans[i].orderline_loc, epoch, result, lane_id);
    }
}

__device__ __forceinline__ void gpuExecTpccTxn(TpccRecords records, TpccVersions versions, PaymentTxnParams *params,
    PaymentTxnExecPlan *plan, uint32_t epoch, uint32_t lane_id, uint32_t txn_id /* for debug TODO: remove*/)
{
    constexpr uint32_t leader_lane = 0;
    constexpr uint32_t all_lanes_mask = 0xffffffffu;
    constexpr uint32_t w_ytd_offset = offsetof(WarehouseValue, w_ytd) / sizeof(uint32_t);
    constexpr uint32_t d_ytd_offset = offsetof(DistrictValue, d_ytd) / sizeof(uint32_t);
    constexpr uint32_t c_balance_offset = offsetof(CustomerValue, c_balance) / sizeof(uint32_t);
    constexpr uint32_t c_ytd_payment_offset = offsetof(CustomerValue, c_ytd_payment) / sizeof(uint32_t);
    constexpr uint32_t c_payment_cnt_offset = offsetof(CustomerValue, c_payment_cnt) / sizeof(uint32_t);

#if 0 // DEBUG
    {
//                if (lane_id == leader_lane && txn_id < 100)
        if (lane_id == leader_lane)
        {
            printf("payment txn_id[%u] warehouse i[%u]rl[%u]wl[%u] district i[%u]rl[%u]wl[%u] customer i[%u]rl[%u]wl[%u] "
                   "payment_amount[%u]\n",
                txn_id, params->warehouse_id, plan->warehouse_read_loc, plan->warehouse_write_loc, params->district_id,
                plan->district_read_loc, plan->district_write_loc, params->customer_id, plan->customer_read_loc,
                plan->customer_write_loc, params->payment_amount);
        }
    }
#endif

    uint32_t result;
    uint32_t payment_amount = params->payment_amount;

    gpuReadFromTableCoop(records.warehouse_record, versions.warehouse_version, params->warehouse_id,
        plan->warehouse_read_loc, epoch, result, lane_id);
    if (lane_id == w_ytd_offset)
    {
        result += payment_amount;
    }
    gpuWriteToTableCoop(records.warehouse_record, versions.warehouse_version, params->warehouse_id,
        plan->warehouse_write_loc, epoch, result, lane_id);

    gpuReadFromTableCoop(records.district_record, versions.district_version, params->district_id,
        plan->district_read_loc, epoch, result, lane_id);
    if (lane_id == d_ytd_offset)
    {
        result += payment_amount;
    }
    gpuWriteToTableCoop(records.district_record, versions.district_version, params->district_id,
        plan->district_write_loc, epoch, result, lane_id);

    gpuReadFromTableCoop(records.customer_record, versions.customer_version, params->customer_id,
        plan->customer_read_loc, epoch, result, lane_id);
    if (lane_id == c_balance_offset)
    {
        result -= payment_amount;
    }
    if (lane_id == c_ytd_payment_offset)
    {
        result += payment_amount;
    }
    if (lane_id == c_payment_cnt_offset)
    {
        result += 1;
    }
    gpuWriteToTableCoop(records.customer_record, versions.customer_version, params->customer_id,
        plan->customer_write_loc, epoch, result, lane_id);
}

__device__ __forceinline__ void gpuExecTpccTxn(TpccRecords records, TpccVersions versions, OrderStatusTxnParams *params,
    OrderStatusTxnExecPlan *plan, uint32_t epoch, uint32_t lane_id, uint32_t txn_id /* for debug TODO: remove*/)
{
    uint32_t result;
    gpuReadFromTableCoop(records.customer_record, versions.customer_version, params->customer_id, plan->customer_loc,
        epoch, result, lane_id);
    gpuReadFromTableCoop(
        records.order_record, versions.order_version, params->order_id, plan->order_loc, epoch, result, lane_id);
    for (int i = 0; i < params->num_items; ++i)
    {
        gpuReadFromTableCoop(records.order_line_record, versions.order_line_version, params->orderline_ids[i],
            plan->orderline_locs[i], epoch, result, lane_id);
    }
}

void __device__ __forceinline__ gpuExecTpccTxn(TpccRecords records, TpccVersions versions, DeliveryTxnParams *params,
    DeliveryTxnExecPlan *plan, uint32_t epoch, uint32_t lane_id, uint32_t txn_id)
{
    uint32_t result;
    for (int i = 0; i < 10; ++i)
    {
        gpuReadFromTableCoop(records.new_order_record, versions.new_order_version, params->new_order_id[i],
            plan->new_order_read_locs[i], epoch, result, lane_id);

        constexpr uint32_t o_carrier_id_offset = offsetof(OrderValue, o_carrier_id) / sizeof(uint32_t);
        gpuReadFromTableCoop(records.order_record, versions.order_version, params->order_id[i],
            plan->order_read_locs[i], epoch, result, lane_id);

        if (lane_id == o_carrier_id_offset)
        {
            result = params->carrier_id;
        }
        gpuWriteToTableCoop(records.order_record, versions.order_version, params->order_id[i],
            plan->order_write_locs[i], epoch, result, lane_id);

        constexpr uint32_t ol_amount_offset = offsetof(OrderLineValue, ol_amount) / sizeof(uint32_t);
        constexpr uint32_t ol_delivery_d_offset = offsetof(OrderLineValue, ol_delivery_d) / sizeof(uint32_t);
        uint32_t amount = 0;
        for (int j = 0; j < params->num_items[i]; ++j)
        {
            gpuReadFromTableCoop(records.order_line_record, versions.order_line_version, params->orderline_ids[i][j],
                plan->orderline_read_locs[i][j], epoch, result, lane_id);
            if (lane_id == ol_amount_offset)
            {
                amount += result;
            }
            if (lane_id == ol_delivery_d_offset)
            {
                result = params->delivery_d;
            }

            gpuWriteToTableCoop(records.order_line_record, versions.order_line_version, params->orderline_ids[i][j],
                loc_record_b, epoch, result, lane_id);
        }

        constexpr uint32_t all_lanes_mask = 0xffffffffu;
        __shfl_sync(all_lanes_mask, amount, ol_amount_offset);

        gpuReadFromTableCoop(records.customer_record, versions.customer_version, params->customer_id[i],
            plan->customer_read_locs[i], epoch, result, lane_id);

        constexpr uint32_t c_balance_offset = offsetof(CustomerValue, c_balance) / sizeof(uint32_t);
        constexpr uint32_t c_delivery_cnt_offset = offsetof(CustomerValue, c_delivery_cnt) / sizeof(uint32_t);

        if (lane_id == c_balance_offset)
        {
            result += amount;
        }
        if (lane_id == c_delivery_cnt_offset)
        {
            ++result;
        }

        gpuWriteToTableCoop(records.customer_record, versions.customer_version, params->customer_id[i],
            plan->customer_write_locs[i], epoch, result, lane_id);

    }
}

void __device__ __forceinline__ gpuExecTpccTxn(TpccRecords records, TpccVersions versions, StockLevelTxnParams *params,
    StockLevelTxnExecPlan *plan, uint32_t epoch, uint32_t lane_id, uint32_t txn_id)
{
    uint32_t num_low_stock = 0;
    const uint32_t threshold = params->threshold;
    uint32_t result;
    constexpr uint32_t s_quantity_offset = offsetof(StockValue, s_quantity) / sizeof(uint32_t);
    for (uint32_t i = 0; i < params->num_items; ++i)
    {
        gpuReadFromTableCoop(records.stock_record, versions.stock_version, params->stock_ids[i],
            plan->stock_read_locs[i], epoch, result, lane_id);
        if (lane_id == s_quantity_offset && result < threshold)
        {
            ++num_low_stock;
        }
    }
    if (lane_id == s_quantity_offset)
    {
        params->num_low_stock = num_low_stock;
    }
}

union CachableTxnParams
{
    NewOrderTxnParams<FixedSizeTxn> no;
    PaymentTxnParams pmt;
    OrderStatusTxnParams os;
} __attribute__((aligned(4)));

union CachableTxnExecPlan
{
    NewOrderExecPlan<FixedSizeTxn> no;
    PaymentTxnExecPlan pmt;
    OrderStatusTxnExecPlan os;
} __attribute__((aligned(4)));

static_assert(sizeof(CachableTxnExecPlan) + sizeof(CachableTxnParams) < 1000);

__global__ void gpuExecKernel(
    TpccRecords records, TpccVersions versions, GpuTxnArray txn, GpuTxnArray plan, uint32_t num_txns, uint32_t epoch)
{
    constexpr uint32_t leader_lane = 0;
    constexpr uint32_t all_lanes_mask = 0xffffffffu;

    __shared__ uint8_t cached_txn_param[num_warps][BaseTxnSize<CachableTxnParams>::value];
    __shared__ uint8_t cached_exec_plan[num_warps][BaseTxnSize<CachableTxnExecPlan>::value];
    static_assert(BaseTxnSize<CachableTxnParams>::value % sizeof(uint32_t) == 0, "Cannot be copied in 32-bit words");
    static_assert(BaseTxnSize<CachableTxnExecPlan>::value % sizeof(uint32_t) == 0, "Cannot be copied in 32-bit words");
    __shared__ uint32_t warp_counter;

    uint32_t warp_id = threadIdx.x / kDeviceWarpSize;
    uint32_t lane_id = threadIdx.x % kDeviceWarpSize;
    /* one thread loads txn id for the entire warp */
    if (threadIdx.x == 0)
    {
        warp_counter = atomicAdd(&txn_counter, num_warps);
    }

    __syncthreads();
    /* warp cooperative execution afterward */

    uint32_t warp_txn_id;
    if (lane_id == leader_lane)
    {
        warp_txn_id = atomicAdd(&warp_counter, 1);
    }
    warp_txn_id = __shfl_sync(all_lanes_mask, warp_txn_id, leader_lane);
    if (warp_txn_id >= num_txns)
    {
        return;
    }

    /* load txn param and exec plan into shared memory */
    BaseTxn *txn_param_ptr = txn.getTxn(warp_txn_id);
    BaseTxn *exec_plan_ptr = plan.getTxn(warp_txn_id);

    /* execute the txn */
    switch (static_cast<TpccTxnType>((reinterpret_cast<BaseTxn *>(txn_param_ptr)->txn_type)))
    {
    case TpccTxnType::NEW_ORDER:
        warpMemcpy(reinterpret_cast<uint32_t *>(cached_txn_param[warp_id]), reinterpret_cast<uint32_t *>(txn_param_ptr),
            BaseTxnSize<NewOrderTxnParams<FixedSizeTxn>>::value / sizeof(uint32_t), lane_id);
        warpMemcpy(reinterpret_cast<uint32_t *>(cached_exec_plan[warp_id]), reinterpret_cast<uint32_t *>(exec_plan_ptr),
            BaseTxnSize<NewOrderExecPlan<FixedSizeTxn>>::value / sizeof(uint32_t), lane_id);
        __syncwarp();
        gpuExecTpccTxn(records, versions,
            reinterpret_cast<NewOrderTxnParams<FixedSizeTxn> *>(
                reinterpret_cast<BaseTxn *>(cached_txn_param[warp_id])->data),
            reinterpret_cast<NewOrderExecPlan<FixedSizeTxn> *>(
                reinterpret_cast<BaseTxn *>(cached_exec_plan[warp_id])->data),
            epoch, lane_id, warp_txn_id);
        break;
    case TpccTxnType::PAYMENT:
        warpMemcpy(reinterpret_cast<uint32_t *>(cached_txn_param[warp_id]), reinterpret_cast<uint32_t *>(txn_param_ptr),
            BaseTxnSize<PaymentTxnParams>::value / sizeof(uint32_t), lane_id);
        warpMemcpy(reinterpret_cast<uint32_t *>(cached_exec_plan[warp_id]), reinterpret_cast<uint32_t *>(exec_plan_ptr),
            BaseTxnSize<PaymentTxnExecPlan>::value / sizeof(uint32_t), lane_id);
        __syncwarp();
        gpuExecTpccTxn(records, versions,
            reinterpret_cast<PaymentTxnParams *>(reinterpret_cast<BaseTxn *>(cached_txn_param[warp_id])->data),
            reinterpret_cast<PaymentTxnExecPlan *>(reinterpret_cast<BaseTxn *>(cached_exec_plan[warp_id])->data), epoch,
            lane_id, warp_txn_id);
        break;
    case TpccTxnType::ORDER_STATUS:
        warpMemcpy(reinterpret_cast<uint32_t *>(cached_txn_param[warp_id]), reinterpret_cast<uint32_t *>(txn_param_ptr),
            BaseTxnSize<OrderStatusTxnParams>::value / sizeof(uint32_t), lane_id);
        warpMemcpy(reinterpret_cast<uint32_t *>(cached_exec_plan[warp_id]), reinterpret_cast<uint32_t *>(exec_plan_ptr),
            BaseTxnSize<OrderStatusTxnExecPlan>::value / sizeof(uint32_t), lane_id);
        __syncwarp();
        gpuExecTpccTxn(records, versions,
            reinterpret_cast<OrderStatusTxnParams *>(reinterpret_cast<BaseTxn *>(cached_txn_param[warp_id])->data),
            reinterpret_cast<OrderStatusTxnExecPlan *>(reinterpret_cast<BaseTxn *>(cached_exec_plan[warp_id])->data),
            epoch, lane_id, warp_txn_id);
        break;
    case TpccTxnType::DELIVERY:
        gpuExecTpccTxn(records, versions, reinterpret_cast<DeliveryTxnParams *>(txn_param_ptr->data),
            reinterpret_cast<DeliveryTxnExecPlan *>(exec_plan_ptr->data), epoch, lane_id, warp_txn_id);
            break;
    case TpccTxnType::STOCK_LEVEL:
        gpuExecTpccTxn(records, versions, reinterpret_cast<StockLevelTxnParams *>(txn_param_ptr->data),
            reinterpret_cast<StockLevelTxnExecPlan *>(exec_plan_ptr->data), epoch, lane_id, warp_txn_id);
        break;
    default:
        assert(false);
        break;
    }
}

} // namespace

void GpuExecutor::execute(uint32_t epoch)
{
    /* clear the txn_counter */
    gpu_err_check(cudaMemcpyToSymbol(txn_counter, &zero, sizeof(uint32_t)));

#if 0 // DEBUG
    {
        auto &logger = Logger::GetInstance();
        constexpr size_t max_print_size = 100u;
        constexpr size_t base_txn_size = TxnArray<TpccTxnParam>::kBaseTxnSize;
        uint32_t print_size = std::min(config.num_txns, max_print_size);
        uint32_t copy_size = print_size * base_txn_size;
        uint8_t txn_params[max_print_size * base_txn_size];

        transferGpuToCpu(txn_params, txn.txns, copy_size);
        for (int i = 0; i < print_size; ++i)
        {
            auto param = &reinterpret_cast<TpccTxnParam *>(
                reinterpret_cast<BaseTxn *>(txn_params + i * base_txn_size)->data)
                ->new_order_txn;
            logger.Info("txn {} warehouse[{}] district[{}] customer[{}] order[{}] new_order[{}] numitems[{}] "
                        "item1[{}] stock_read1[{}] order_line1[{}] quantity1[{}] "
                        "item2[{}] stock_read2[{}] order_line2[{}] quantity2[{}] "
                        "item3[{}] stock_read3[{}] order_line3[{}] quantity3[{}] "
                        "item4[{}] stock_read4[{}] order_line4[{}] quantity4[{}] "
                        "item5[{}] stock_read5[{}] order_line5[{}] quantity5[{}] ",
                        i, param->warehouse_id, param->district_id, param->customer_id, param->order_id,
                        param->new_order_id, param->num_items, param->items[0].item_id, param->items[0].stock_id,
                        param->items[0].order_line_id, param->items[0].order_quantities, param->items[1].item_id,
                        param->items[1].stock_id, param->items[1].order_line_id, param->items[1].order_quantities,
                        param->items[2].item_id, param->items[2].stock_id, param->items[2].order_line_id,
                        param->items[2].order_quantities, param->items[3].item_id, param->items[3].stock_id,
                        param->items[3].order_line_id, param->items[3].order_quantities, param->items[4].item_id,
                        param->items[4].stock_id, param->items[4].order_line_id, param->items[4].order_quantities);
        }
        logger.flush();
    }
#endif

    uint32_t num_blocks = (config.num_txns * kDeviceWarpSize + block_size - 1) / block_size;
    gpuExecKernel<<<num_blocks, block_size>>>(
        records, versions, GpuTxnArray(txn), GpuTxnArray(plan), config.num_txns, epoch);
    gpu_err_check(cudaPeekAtLastError());
    gpu_err_check(cudaDeviceSynchronize());
}

} // namespace epic::tpcc