//
// Created by Shujian Qian on 2023-11-10.
//

#include <gacco/benchmarks/tpcc_gpu_executor.h>
#include <cstdio>

#include <util_gpu_error_check.cuh>
#include <gpu_txn.cuh>

namespace gacco::tpcc {

using epic::BaseTxn;
using epic::GpuTxnArray;

namespace {

__device__ uint32_t txn_counter = 0; /* used for scheduling txns among threads */
const uint32_t zero = 0;

__device__ __forceinline__ void acquireLock(GaccoTableLock lock, uint32_t record_id, uint32_t txn_id)
{
    bool printed = false;
    uint32_t current_lock_holder = atomicAdd(&lock.lock[record_id], 0);
    //    if (txn_id < 32) printf("txn %u current_holder is %u\n", txn_id, current_lock_holder);
    while (current_lock_holder != txn_id)
    {
        //        if (txn_id < 32) printf("txn %u waiting for lock, current_holder is %u\n", txn_id,
        //        current_lock_holder);
        /* spin */
        if (!printed)
        {
            //            if (txn_id == 0) printf("txn %u waiting for lock, current_holder is %u\n", txn_id,
            //            current_lock_holder); printf("txn %u waiting for lock, current_holder is %u\n", txn_id,
            //            current_lock_holder);
            printed = true;
        }
        current_lock_holder = atomicAdd(&lock.lock[record_id], 0);
    }
    //    if (txn_id == 0) printf("hello current_holder is %u acquire successful\n", current_lock_holder);
}

__device__ __forceinline__ void releaseLock(GaccoTableLock lock, uint32_t record_id, uint32_t txn_id)
{
    uint32_t new_lock_offset = atomicAdd(&lock.offset[record_id], 1) + 1;
    __threadfence();
    uint32_t new_lock_holder = atomicAdd(&lock.access[new_lock_offset], 0);
    atomicExch(&lock.lock[record_id], new_lock_holder);
    __threadfence();
    //    if (txn_id == 0) printf("hello released successful next_holder %u\n", new_lock_holder);
    //    printf("txn %u released lock next holder is %u\n", txn_id, new_lock_holder);
}

__device__ __forceinline__ void readData(uint32_t *data, uint32_t size, uint32_t &output)
{
    for (uint32_t i = 0; i < size; i++)
    {
        output += data[i];
    }
}
__device__ __forceinline__ void writeData(uint32_t *data, uint32_t size, uint32_t input)
{
    for (uint32_t i = 0; i < size; i++)
    {
        data[i] = input;
        --input;
    }
}

__device__ __forceinline__ void gpuExecTpccTxn(epic::tpcc::TpccConfig config, TpccRecords records,
    Executor::TpccTableLocks table_locks, NewOrderTxnParams<FixedSizeTxn> *params, uint32_t txn_id)
{
    uint32_t data = 0;
    if (config.gacco_use_atomic)
    {
        WarehouseValue *warehouse_value = &records.warehouse_record[params->warehouse_id];
        readData(reinterpret_cast<uint32_t *>(warehouse_value), sizeof(WarehouseValue) / sizeof(uint32_t), data);
        DistrictValue *district_value = &records.district_record[params->district_id];
        readData(reinterpret_cast<uint32_t *>(district_value), sizeof(DistrictValue) / sizeof(uint32_t), data);
        CustomerValue *customer_value = &records.customer_record[params->customer_id];
        readData(reinterpret_cast<uint32_t *>(customer_value), sizeof(CustomerValue) / sizeof(uint32_t), data);

        NewOrderValue *new_order_value = &records.new_order_record[params->new_order_id];
        writeData(reinterpret_cast<uint32_t *>(new_order_value), sizeof(NewOrderValue) / sizeof(uint32_t), data);
        OrderValue *order_value = &records.order_record[params->order_id];
        writeData(reinterpret_cast<uint32_t *>(order_value), sizeof(OrderValue) / sizeof(uint32_t), data);

        for (int i = 0; i < params->num_items; ++i)
        {
            OrderLineValue *order_line_value = &records.order_line_record[params->items[i].order_line_id];
            writeData(reinterpret_cast<uint32_t *>(order_line_value), sizeof(OrderLineValue) / sizeof(uint32_t), data);

            StockValue *stock_value = &records.stock_record[params->items[i].stock_id];
            if (config.gacco_tpcc_stock_use_atomic)
            {
                readData(reinterpret_cast<uint32_t *>(stock_value), sizeof(StockValue) / sizeof(uint32_t), data);
                uint32_t assumed_quantity, adjusted_quantity, old_quantity;
                old_quantity = stock_value->s_quantity;
                do
                {
                    assumed_quantity = old_quantity;
                    uint32_t order_quantity = params->items[i].order_quantities;
                    adjusted_quantity = assumed_quantity >= order_quantity + 10
                                            ? assumed_quantity - order_quantity
                                            : assumed_quantity + 91 - order_quantity;
                } while ((old_quantity = atomicCAS(&stock_value->s_quantity, assumed_quantity, adjusted_quantity)) !=
                         assumed_quantity);
            }
            else
            {
                acquireLock(table_locks.stock, params->items[i].stock_id, txn_id);
                readData(reinterpret_cast<uint32_t *>(stock_value), sizeof(StockValue) / sizeof(uint32_t), data);
                uint32_t order_quantity = params->items[i].order_quantities;
                uint32_t old_quantity = stock_value->s_quantity;
                stock_value->s_quantity = old_quantity >= order_quantity + 10 ? old_quantity - order_quantity
                                                                              : old_quantity + 91 - order_quantity;
                __threadfence();
                releaseLock(table_locks.stock, params->items[i].stock_id, txn_id);
            }

            ItemValue *item_value = &records.item_record[params->items[i].item_id];
            readData(reinterpret_cast<uint32_t *>(item_value), sizeof(ItemValue) / sizeof(uint32_t), data);
        }
    }
    else
    {
        WarehouseValue *warehouse_value = &records.warehouse_record[params->warehouse_id];
        DistrictValue *district_value = &records.district_record[params->district_id];
        CustomerValue *customer_value = &records.customer_record[params->customer_id];

        NewOrderValue *new_order_value = &records.new_order_record[params->new_order_id];
        OrderValue *order_value = &records.order_record[params->order_id];

        acquireLock(table_locks.warehouse, params->warehouse_id, txn_id);
        readData(reinterpret_cast<uint32_t *>(warehouse_value), sizeof(WarehouseValue) / sizeof(uint32_t), data);
        releaseLock(table_locks.warehouse, params->warehouse_id, txn_id);

        acquireLock(table_locks.district, params->district_id, txn_id);
        readData(reinterpret_cast<uint32_t *>(district_value), sizeof(DistrictValue) / sizeof(uint32_t), data);
        releaseLock(table_locks.district, params->district_id, txn_id);

        acquireLock(table_locks.customer, params->customer_id, txn_id);
        readData(reinterpret_cast<uint32_t *>(customer_value), sizeof(CustomerValue) / sizeof(uint32_t), data);
        releaseLock(table_locks.customer, params->customer_id, txn_id);

        acquireLock(table_locks.new_order, params->new_order_id, txn_id);
        acquireLock(table_locks.order, params->order_id, txn_id);
        writeData(reinterpret_cast<uint32_t *>(new_order_value), sizeof(NewOrderValue) / sizeof(uint32_t), data);
        writeData(reinterpret_cast<uint32_t *>(order_value), sizeof(OrderValue) / sizeof(uint32_t), data);
        __threadfence();
        releaseLock(table_locks.new_order, params->new_order_id, txn_id);
        releaseLock(table_locks.order, params->order_id, txn_id);

        for (uint32_t i = 0; i < params->num_items; i++)
        {
            OrderLineValue *order_line_value = &records.order_line_record[params->items[i].order_line_id];
            StockValue *stock_value = &records.stock_record[params->items[i].stock_id];

            acquireLock(table_locks.order_line, params->items[i].order_line_id, txn_id);
            writeData(reinterpret_cast<uint32_t *>(order_line_value), sizeof(OrderLineValue) / sizeof(uint32_t), data);
            acquireLock(table_locks.stock, params->items[i].stock_id, txn_id);
            writeData(reinterpret_cast<uint32_t *>(stock_value), sizeof(StockValue) / sizeof(uint32_t), data);
            __threadfence();
            releaseLock(table_locks.order_line, params->items[i].order_line_id, txn_id);
            releaseLock(table_locks.stock, params->items[i].stock_id, txn_id);

            ItemValue *item_value = &records.item_record[params->items[i].item_id];
            acquireLock(table_locks.item, params->items[i].item_id, txn_id);
            readData(reinterpret_cast<uint32_t *>(item_value), sizeof(ItemValue) / sizeof(uint32_t), data);
            releaseLock(table_locks.item, params->items[i].item_id, txn_id);
        }
    }
    params->warehouse_id = data; /* to prevent compiler from optimizing out the txn */
}

__device__ __forceinline__ void gpuExecTpccTxn(epic::tpcc::TpccConfig config, TpccRecords records,
    Executor::TpccTableLocks table_locks, PaymentTxnParams *params, uint32_t txn_id)
{
    WarehouseValue *warehouse_value = &records.warehouse_record[params->warehouse_id];
    DistrictValue *district_value = &records.district_record[params->district_id];
    CustomerValue *customer_value = &records.customer_record[params->customer_id];
    uint32_t data = 0;
    if (config.gacco_use_atomic)
    {
        /* execute without holding locks */
        readData(reinterpret_cast<uint32_t *>(warehouse_value), sizeof(WarehouseValue) / sizeof(uint32_t), data);
        atomicAdd(&warehouse_value->w_ytd, params->payment_amount);

        readData(reinterpret_cast<uint32_t *>(customer_value), sizeof(CustomerValue) / sizeof(uint32_t), data);
        atomicAdd(&district_value->d_ytd, params->payment_amount);

        readData(reinterpret_cast<uint32_t *>(customer_value), sizeof(CustomerValue) / sizeof(uint32_t), data);
        atomicSub(&customer_value->c_balance, params->payment_amount);
        atomicAdd(&customer_value->c_ytd_payment, params->payment_amount);
        atomicAdd(&customer_value->c_payment_cnt, 1);
    }
    else
    {
        acquireLock(table_locks.warehouse, params->warehouse_id, txn_id);
        readData(reinterpret_cast<uint32_t *>(customer_value), sizeof(CustomerValue) / sizeof(uint32_t), data);
        warehouse_value->w_ytd += params->payment_amount;

        acquireLock(table_locks.district, params->district_id, txn_id);
        readData(reinterpret_cast<uint32_t *>(customer_value), sizeof(CustomerValue) / sizeof(uint32_t), data);
        district_value->d_ytd += params->payment_amount;

        acquireLock(table_locks.customer, params->customer_id, txn_id);
        readData(reinterpret_cast<uint32_t *>(customer_value), sizeof(CustomerValue) / sizeof(uint32_t), data);
        customer_value->c_balance -= params->payment_amount;
        customer_value->c_ytd_payment += params->payment_amount;
        customer_value->c_payment_cnt++;
        __threadfence();
        releaseLock(table_locks.warehouse, params->warehouse_id, txn_id);
        releaseLock(table_locks.district, params->district_id, txn_id);
        releaseLock(table_locks.customer, params->customer_id, txn_id);
    }
    params->warehouse_id = data; /* to prevent compiler from optimizing out the txn */
}

__global__ void gpuExecKernel(epic::tpcc::TpccConfig config, TpccRecords records, Executor::TpccTableLocks table_locks,
    epic::GpuPackedTxnArray txn, uint32_t num_txns)
{
    __shared__ uint32_t block_counter;
    /* one thread loads txn id for the entire warp */
    uint32_t block_size = blockDim.x;
    uint32_t tid_in_block = threadIdx.x;
    if (threadIdx.x == 0)
    {
        block_counter = atomicAdd(&txn_counter, block_size);
    }
    __syncthreads();

    uint32_t txn_id = block_counter + tid_in_block;
    if (txn_id >= num_txns)
    {
        return;
    }

    BaseTxn *txn_param_ptr = txn.getTxn(txn_id);

    /* execute the txn */

    switch (static_cast<TpccTxnType>(txn_param_ptr->txn_type))
    {
    case TpccTxnType::NEW_ORDER:
        gpuExecTpccTxn(config, records, table_locks,
            reinterpret_cast<NewOrderTxnParams<FixedSizeTxn> *>(txn_param_ptr->data), txn_id);
        break;
    case TpccTxnType::PAYMENT:
        gpuExecTpccTxn(config, records, table_locks, reinterpret_cast<PaymentTxnParams *>(txn_param_ptr->data), txn_id);
        break;
    default:
        assert(false);
        break;
    }
//            if ((txn_id + 1) % 100 == 0) {
//                printf("txn %u finished\n", txn_id + 1);
//            }

    //    for (int i = 0; i < 32; ++i) {
    //        if (threadIdx.x % 32 == i) {
    //            switch (static_cast<TpccTxnType>(txn_param_ptr->txn_type))
    //            {
    //            case TpccTxnType::NEW_ORDER:
    //                gpuExecTpccTxn(table_locks, reinterpret_cast<NewOrderTxnParams<FixedSizeTxn>
    //                *>(txn_param_ptr->data), txn_id); break;
    //            case TpccTxnType::PAYMENT:
    //                gpuExecTpccTxn(table_locks, reinterpret_cast<PaymentTxnParams *>(txn_param_ptr->data), txn_id);
    //                break;
    //            default:
    //                assert(false);
    //                break;
    //            }
    //        }
    //        __syncwarp();
    //    }
}

} // namespace

void GpuExecutor::execute(uint32_t epoch)
{
    /* clear the txn_counter */
    gpu_err_check(cudaMemcpyToSymbol(txn_counter, &zero, sizeof(uint32_t)));

    constexpr uint32_t block_size = 256;

    gpuExecKernel<<<(config.num_txns + block_size - 1) / block_size, block_size>>>(
        config, records, table_locks, epic::GpuPackedTxnArray(txn), config.num_txns);
    gpu_err_check(cudaPeekAtLastError());
    gpu_err_check(cudaDeviceSynchronize());
}

} // namespace gacco::tpcc