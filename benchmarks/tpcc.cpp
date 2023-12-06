//
// Created by Shujian Qian on 2023-09-15.
//

#include "benchmarks/tpcc.h"

#include <cassert>
#include <random>
#include <cstdio>
#include <chrono>

#include "benchmarks/tpcc_txn.h"

#include "util_log.h"
#include "util_device_type.h"
#include "gpu_txn.h"
#include "gpu_allocator.h"
#include "gpu_execution_planner.h"
#include "benchmarks/tpcc_gpu_submitter.h"
#include "benchmarks/tpcc_executor.h"
#include "benchmarks/tpcc_gpu_executor.h"
#include <benchmarks/tpcc_txn_gen.h>
#include <benchmarks/tpcc_gpu_index.h>
#include <benchmarks/tpcc_cpu_executor.h>

namespace epic::tpcc {
TpccTxnMix::TpccTxnMix(
    uint32_t new_order, uint32_t payment, uint32_t order_status, uint32_t delivery, uint32_t stock_level)
    : new_order(new_order)
    , payment(payment)
    , order_status(order_status)
    , delivery(delivery)
    , stock_level(stock_level)
{
    assert(new_order + payment + order_status + delivery + stock_level == 100);
}

TpccDb::TpccDb(TpccConfig config)
    : config(config)
    , txn_array(config.epochs)
    , index_input(config.num_txns, config.index_device, false)
    , index_output(config.num_txns, config.index_device)
    , initialization_input(config.num_txns, config.initialize_device, false)
    , initialization_output(config.num_txns, config.initialize_device)
    , execution_param_input(config.num_txns, config.execution_device, false)
    , execution_plan_input(config.num_txns, config.execution_device, false)
{
    //    index = std::make_shared<TpccCpuIndex>(config);
    for (int i = 0; i < config.epochs; ++i)
    {
        txn_array[i] = TxnArray<TpccTxn>(config.num_txns, DeviceType::CPU);
    }
    if (config.index_device == DeviceType::CPU)
    {
        index = std::make_shared<TpccCpuIndex>(config);
    }
    else if (config.index_device == DeviceType::GPU)
    {
        index = std::make_shared<TpccGpuIndex>(config);
    }
    else
    {
        throw std::runtime_error("Unsupported index device");
    }

    input_index_bridge.Link(txn_array[0], index_input);
    index_initialization_bridge.Link(index_output, initialization_input);
    index_execution_param_bridge.Link(index_output, execution_param_input);
    initialization_execution_plan_bridge.Link(initialization_output, execution_plan_input);

    if (config.initialize_device == DeviceType::GPU)
    {
        GpuAllocator allocator;
        warehouse_planner = std::make_unique<GpuTableExecutionPlanner>(
            "warehouse", allocator, 0, 2, config.num_txns, config.num_warehouses, initialization_output);
        district_planner = std::make_unique<GpuTableExecutionPlanner>(
            "district", allocator, 0, 2, config.num_txns, config.num_warehouses * 10, initialization_output);
        customer_planner = std::make_unique<GpuTableExecutionPlanner>(
            "customer", allocator, 0, 2, config.num_txns, config.num_warehouses * 10 * 3000, initialization_output);
        history_planner = std::make_unique<GpuTableExecutionPlanner>(
            "history", allocator, 0, 1, config.num_txns, config.num_warehouses * 10 * 3000, initialization_output);
        new_order_planner = std::make_unique<GpuTableExecutionPlanner>(
            "new_order", allocator, 0, 1, config.num_txns, config.num_warehouses * 10 * 900, initialization_output);
        order_planner = std::make_unique<GpuTableExecutionPlanner>(
            "order", allocator, 0, 1, config.num_txns, config.num_warehouses * 10 * 3000, initialization_output);
        order_line_planner = std::make_unique<GpuTableExecutionPlanner>("order_line", allocator, 0, 15, config.num_txns,
            config.num_warehouses * 10 * 3000 * 15, initialization_output);
        item_planner = std::make_unique<GpuTableExecutionPlanner>(
            "item", allocator, 0, 15, config.num_txns, 100'000, initialization_output);
        stock_planner = std::make_unique<GpuTableExecutionPlanner>(
            "stock", allocator, 0, 15 * 2, config.num_txns, 100'000 * config.num_warehouses, initialization_output);

        warehouse_planner->Initialize();
        district_planner->Initialize();
        customer_planner->Initialize();
        history_planner->Initialize();
        new_order_planner->Initialize();
        order_planner->Initialize();
        order_line_planner->Initialize();
        item_planner->Initialize();
        stock_planner->Initialize();
        allocator.PrintMemoryInfo();

        submitter = std::make_shared<TpccGpuSubmitter>(
            TpccSubmitter::TableSubmitDest{warehouse_planner->d_num_ops, warehouse_planner->d_op_offsets,
                warehouse_planner->d_submitted_ops, warehouse_planner->d_scratch_array,
                warehouse_planner->scratch_array_bytes, warehouse_planner->curr_num_ops},
            TpccSubmitter::TableSubmitDest{district_planner->d_num_ops, district_planner->d_op_offsets,
                district_planner->d_submitted_ops, district_planner->d_scratch_array,
                district_planner->scratch_array_bytes, district_planner->curr_num_ops},
            TpccSubmitter::TableSubmitDest{customer_planner->d_num_ops, customer_planner->d_op_offsets,
                customer_planner->d_submitted_ops, customer_planner->d_scratch_array,
                customer_planner->scratch_array_bytes, customer_planner->curr_num_ops},
            TpccSubmitter::TableSubmitDest{history_planner->d_num_ops, history_planner->d_op_offsets,
                history_planner->d_submitted_ops, history_planner->d_scratch_array,
                history_planner->scratch_array_bytes, history_planner->curr_num_ops},
            TpccSubmitter::TableSubmitDest{new_order_planner->d_num_ops, new_order_planner->d_op_offsets,
                new_order_planner->d_submitted_ops, new_order_planner->d_scratch_array,
                new_order_planner->scratch_array_bytes, new_order_planner->curr_num_ops},
            TpccSubmitter::TableSubmitDest{order_planner->d_num_ops, order_planner->d_op_offsets,
                order_planner->d_submitted_ops, order_planner->d_scratch_array, order_planner->scratch_array_bytes,
                order_planner->curr_num_ops},
            TpccSubmitter::TableSubmitDest{order_line_planner->d_num_ops, order_line_planner->d_op_offsets,
                order_line_planner->d_submitted_ops, order_line_planner->d_scratch_array,
                order_line_planner->scratch_array_bytes, order_line_planner->curr_num_ops},
            TpccSubmitter::TableSubmitDest{item_planner->d_num_ops, item_planner->d_op_offsets,
                item_planner->d_submitted_ops, item_planner->d_scratch_array, item_planner->scratch_array_bytes,
                item_planner->curr_num_ops},
            TpccSubmitter::TableSubmitDest{stock_planner->d_num_ops, stock_planner->d_op_offsets,
                stock_planner->d_submitted_ops, stock_planner->d_scratch_array, stock_planner->scratch_array_bytes,
                stock_planner->curr_num_ops});
    }
    else
    {
        auto &logger = Logger::GetInstance();
        logger.Error("Unsupported initialize device");
        exit(-1);
    }

    if (config.execution_device == DeviceType::GPU)
    {
        /* TODO: initialize records & versions */
        auto &logger = Logger::GetInstance();
        logger.Info("Allocating records and versions");

        GpuAllocator allocator;

        /* CAUTION: version size is based on the number of transactions, and will cause sync issue if too small */
        size_t warehouse_rec_size = sizeof(Record<WarehouseValue>) * config.warehouseTableSize();
        size_t warehouse_ver_size = sizeof(Version<WarehouseValue>) * config.num_txns;
        logger.Info("Warehouse record: {}, version: {}", formatSizeBytes(warehouse_rec_size),
            formatSizeBytes(warehouse_ver_size));
        records.warehouse_record = static_cast<Record<WarehouseValue> *>(allocator.Allocate(warehouse_rec_size));
        versions.warehouse_version = static_cast<Version<WarehouseValue> *>(allocator.Allocate(warehouse_ver_size));

        size_t district_rec_size = sizeof(Record<DistrictValue>) * config.districtTableSize();
        size_t district_ver_size = sizeof(Version<DistrictValue>) * config.num_txns;
        logger.Info(
            "District record: {}, version: {}", formatSizeBytes(district_rec_size), formatSizeBytes(district_ver_size));
        records.district_record = static_cast<Record<DistrictValue> *>(allocator.Allocate(district_rec_size));
        versions.district_version = static_cast<Version<DistrictValue> *>(allocator.Allocate(district_ver_size));

        size_t customer_rec_size = sizeof(Record<CustomerValue>) * config.customerTableSize();
        size_t customer_ver_size = sizeof(Version<CustomerValue>) * config.num_txns;
        logger.Info(
            "Customer record: {}, version: {}", formatSizeBytes(customer_rec_size), formatSizeBytes(customer_ver_size));
        records.customer_record = static_cast<Record<CustomerValue> *>(allocator.Allocate(customer_rec_size));
        versions.customer_version = static_cast<Version<CustomerValue> *>(allocator.Allocate(customer_ver_size));

        /* TODO: history table is too big */
        //        size_t history_rec_size = sizeof(Record<HistoryValue>) * config.historyTableSize();
        //        size_t history_ver_size = sizeof(Version<HistoryValue>) * config.historyTableSize();
        //        logger.Info("History record: {}, version: {}", formatSizeBytes(history_rec_size),
        //                    formatSizeBytes(history_ver_size));
        //        records.history_record = static_cast<Record<HistoryValue> *>(allocator.Allocate(history_rec_size));
        //        versions.history_version = static_cast<Version<HistoryValue> *>(allocator.Allocate(history_ver_size));

        size_t new_order_rec_size = sizeof(Record<NewOrderValue>) * config.newOrderTableSize();
        size_t new_order_ver_size = sizeof(Version<NewOrderValue>) * config.num_txns; /* TODO: not needed */
        logger.Info("NewOrder record: {}, version: {}", formatSizeBytes(new_order_rec_size),
            formatSizeBytes(new_order_ver_size));
        records.new_order_record = static_cast<Record<NewOrderValue> *>(allocator.Allocate(new_order_rec_size));
        versions.new_order_version = static_cast<Version<NewOrderValue> *>(allocator.Allocate(new_order_ver_size));

        size_t order_rec_size = sizeof(Record<OrderValue>) * config.orderTableSize();
        size_t order_ver_size = sizeof(Version<OrderValue>) * config.num_txns; /* TODO: not needed */
        logger.Info("Order record: {}, version: {}", formatSizeBytes(order_rec_size), formatSizeBytes(order_ver_size));
        records.order_record = static_cast<Record<OrderValue> *>(allocator.Allocate(order_rec_size));
        versions.order_version = static_cast<Version<OrderValue> *>(allocator.Allocate(order_ver_size));

        size_t order_line_rec_size = sizeof(Record<OrderLineValue>) * config.orderLineTableSize();
        size_t order_line_ver_size = sizeof(Version<OrderLineValue>) * config.num_txns * 15; /* TODO: not needed */
        logger.Info("OrderLine record: {}, version: {}", formatSizeBytes(order_line_rec_size),
            formatSizeBytes(order_line_ver_size));
        records.order_line_record = static_cast<Record<OrderLineValue> *>(allocator.Allocate(order_line_rec_size));
        versions.order_line_version = static_cast<Version<OrderLineValue> *>(allocator.Allocate(order_line_ver_size));

        size_t item_rec_size = sizeof(Record<ItemValue>) * config.itemTableSize();
        size_t item_ver_size = sizeof(Version<ItemValue>) * config.num_txns * 15; /* TODO: not needed */
        logger.Info("Item record: {}, version: {}", formatSizeBytes(item_rec_size), formatSizeBytes(item_ver_size));
        records.item_record = static_cast<Record<ItemValue> *>(allocator.Allocate(item_rec_size));
        versions.item_version = static_cast<Version<ItemValue> *>(allocator.Allocate(item_ver_size));

        size_t stock_rec_size = sizeof(Record<StockValue>) * config.stockTableSize();
        size_t stock_ver_size = sizeof(Version<StockValue>) * config.num_txns * 15;
        logger.Info("Stock record: {}, version: {}", formatSizeBytes(stock_rec_size), formatSizeBytes(stock_ver_size));
        records.stock_record = static_cast<Record<StockValue> *>(allocator.Allocate(stock_rec_size));
        versions.stock_version = static_cast<Version<StockValue> *>(allocator.Allocate(stock_ver_size));

        allocator.PrintMemoryInfo();

        /* TODO: execution input need to be transferred too, currently using placeholders */
//        executor =
//            std::make_shared<GpuExecutor>(records, versions, initialization_input, initialization_output, config);
        executor =
            std::make_shared<GpuExecutor>(records, versions, execution_param_input, execution_plan_input, config);
    }
    else if (config.execution_device == DeviceType::CPU)
    {
        auto &logger = Logger::GetInstance();
        logger.Info("Allocating records and versions");

        /* CAUTION: version size is based on the number of transactions, and will cause sync issue if too small */
        size_t warehouse_rec_size = sizeof(Record<WarehouseValue>) * config.warehouseTableSize();
        size_t warehouse_ver_size = sizeof(Version<WarehouseValue>) * config.num_txns;
        logger.Info("Warehouse record: {}, version: {}", formatSizeBytes(warehouse_rec_size),
                    formatSizeBytes(warehouse_ver_size));
        records.warehouse_record = static_cast<Record<WarehouseValue> *>(Malloc(warehouse_rec_size));
        versions.warehouse_version = static_cast<Version<WarehouseValue> *>(Malloc(warehouse_ver_size));

        size_t district_rec_size = sizeof(Record<DistrictValue>) * config.districtTableSize();
        size_t district_ver_size = sizeof(Version<DistrictValue>) * config.num_txns;
        logger.Info(
            "District record: {}, version: {}", formatSizeBytes(district_rec_size), formatSizeBytes(district_ver_size));
        records.district_record = static_cast<Record<DistrictValue> *>(Malloc(district_rec_size));
        versions.district_version = static_cast<Version<DistrictValue> *>(Malloc(district_ver_size));

        size_t customer_rec_size = sizeof(Record<CustomerValue>) * config.customerTableSize();
        size_t customer_ver_size = sizeof(Version<CustomerValue>) * config.num_txns;
        logger.Info(
            "Customer record: {}, version: {}", formatSizeBytes(customer_rec_size), formatSizeBytes(customer_ver_size));
        records.customer_record = static_cast<Record<CustomerValue> *>(Malloc(customer_rec_size));
        versions.customer_version = static_cast<Version<CustomerValue> *>(Malloc(customer_ver_size));

        /* TODO: history table is too big */
        //        size_t history_rec_size = sizeof(Record<HistoryValue>) * config.historyTableSize();
        //        size_t history_ver_size = sizeof(Version<HistoryValue>) * config.historyTableSize();
        //        logger.Info("History record: {}, version: {}", formatSizeBytes(history_rec_size),
        //                    formatSizeBytes(history_ver_size));
        //        records.history_record = static_cast<Record<HistoryValue> *>(Malloc(history_rec_size));
        //        versions.history_version = static_cast<Version<HistoryValue> *>(Malloc(history_ver_size));

        size_t new_order_rec_size = sizeof(Record<NewOrderValue>) * config.newOrderTableSize();
        size_t new_order_ver_size = sizeof(Version<NewOrderValue>) * config.num_txns; /* TODO: not needed */
        logger.Info("NewOrder record: {}, version: {}", formatSizeBytes(new_order_rec_size),
                    formatSizeBytes(new_order_ver_size));
        records.new_order_record = static_cast<Record<NewOrderValue> *>(Malloc(new_order_rec_size));
        versions.new_order_version = static_cast<Version<NewOrderValue> *>(Malloc(new_order_ver_size));

        size_t order_rec_size = sizeof(Record<OrderValue>) * config.orderTableSize();
        size_t order_ver_size = sizeof(Version<OrderValue>) * config.num_txns; /* TODO: not needed */
        logger.Info("Order record: {}, version: {}", formatSizeBytes(order_rec_size), formatSizeBytes(order_ver_size));
        records.order_record = static_cast<Record<OrderValue> *>(Malloc(order_rec_size));
        versions.order_version = static_cast<Version<OrderValue> *>(Malloc(order_ver_size));

        size_t order_line_rec_size = sizeof(Record<OrderLineValue>) * config.orderLineTableSize();
        size_t order_line_ver_size = sizeof(Version<OrderLineValue>) * config.num_txns * 15; /* TODO: not needed */
        logger.Info("OrderLine record: {}, version: {}", formatSizeBytes(order_line_rec_size),
                    formatSizeBytes(order_line_ver_size));
        records.order_line_record = static_cast<Record<OrderLineValue> *>(Malloc(order_line_rec_size));
        versions.order_line_version = static_cast<Version<OrderLineValue> *>(Malloc(order_line_ver_size));

        size_t item_rec_size = sizeof(Record<ItemValue>) * config.itemTableSize();
        size_t item_ver_size = sizeof(Version<ItemValue>) * config.num_txns * 15; /* TODO: not needed */
        logger.Info("Item record: {}, version: {}", formatSizeBytes(item_rec_size), formatSizeBytes(item_ver_size));
        records.item_record = static_cast<Record<ItemValue> *>(Malloc(item_rec_size));
        versions.item_version = static_cast<Version<ItemValue> *>(Malloc(item_ver_size));

        size_t stock_rec_size = sizeof(Record<StockValue>) * config.stockTableSize();
        size_t stock_ver_size = sizeof(Version<StockValue>) * config.num_txns * 15;
        logger.Info("Stock record: {}, version: {}", formatSizeBytes(stock_rec_size), formatSizeBytes(stock_ver_size));
        records.stock_record = static_cast<Record<StockValue> *>(Malloc(stock_rec_size));
        versions.stock_version = static_cast<Version<StockValue> *>(Malloc(stock_ver_size));
        executor =
            std::make_shared<CpuExecutor>(records, versions, execution_param_input, execution_plan_input, config);
    }
    else
    {
        auto &logger = Logger::GetInstance();
        logger.Error("Unsupported initialize device");
        exit(-1);
    }
}

void TpccDb::generateTxns()
{
    auto &logger = Logger::GetInstance();

    TpccTxnGenerator generator(config);
    for (size_t epoch = 0; epoch < config.epochs; ++epoch)
    {
        logger.Info("Generating epoch {}", epoch);
        for (size_t i = 0; i < config.num_txns; ++i)
        {
            BaseTxn *txn = txn_array[epoch].getTxn(i);
            uint32_t timestamp = epoch * config.num_txns + i;
            generator.generateTxn(txn, timestamp);
        }
    }
}

void TpccDb::loadInitialData()
{
    index->loadInitialData();
}

void TpccDb::runBenchmark()
{
    auto &logger = Logger::GetInstance();
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    for (uint32_t epoch_id = 1; epoch_id <= config.epochs; ++epoch_id)
    {
        logger.Info("Running epoch {}", epoch_id);
        /* transfer */
        {
            start_time = std::chrono::high_resolution_clock::now();
            uint32_t index_epoch_id = epoch_id - 1;
            input_index_bridge.Link(txn_array[index_epoch_id], index_input);
            input_index_bridge.StartTransfer();
            input_index_bridge.FinishTransfer();

#if 0 // DEBUG
            {
                constexpr size_t max_print_size = 100u;
                constexpr size_t base_txn_size = TxnArray<TpccTxn>::kBaseTxnSize;
                uint32_t print_size = std::min(config.num_txns, max_print_size);
                uint32_t copy_size = print_size * base_txn_size;
                uint8_t txn_params[max_print_size * base_txn_size];

                transferGpuToCpu(txn_params, index_input.txns, copy_size);

                for (int i = 0; i < print_size; ++i)
                {
                    auto param = reinterpret_cast<BaseTxn *>(txn_params + i * base_txn_size)->txn_type;
                    logger.Info("txn {} type {}", i, param);
                }
                logger.flush();
            }

#endif
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} index_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* index */
        {
            start_time = std::chrono::high_resolution_clock::now();
            uint32_t index_epoch_id = epoch_id - 1;
            index->indexTxns(index_input, index_output, index_epoch_id);
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} indexing time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* transfer */
        {
            start_time = std::chrono::high_resolution_clock::now();
            index_initialization_bridge.StartTransfer();
            index_initialization_bridge.FinishTransfer();
            index_execution_param_bridge.StartTransfer();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} init_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());

#if 0 // DEBUG
            {
                constexpr size_t max_print_size = 100u;
                constexpr size_t base_txn_size = TxnArray<TpccTxnParam>::kBaseTxnSize;
                uint32_t print_size = std::min(config.num_txns, max_print_size);
                uint32_t copy_size = print_size * base_txn_size;
                uint8_t txn_params[max_print_size * base_txn_size];

                transferGpuToCpu(txn_params, initialization_input.txns, copy_size);
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
        }

        /* submit */
        {
            start_time = std::chrono::high_resolution_clock::now();
            submitter->submit(initialization_input);
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} submission time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* initialize */
        {
            start_time = std::chrono::high_resolution_clock::now();

            warehouse_planner->InitializeExecutionPlan();
            district_planner->InitializeExecutionPlan();
            customer_planner->InitializeExecutionPlan();
            history_planner->InitializeExecutionPlan();
            new_order_planner->InitializeExecutionPlan();
            order_planner->InitializeExecutionPlan();
            order_line_planner->InitializeExecutionPlan();
            item_planner->InitializeExecutionPlan();
            stock_planner->InitializeExecutionPlan();

            warehouse_planner->FinishInitialization();
            district_planner->FinishInitialization();
            customer_planner->FinishInitialization();
            history_planner->FinishInitialization();
            new_order_planner->FinishInitialization();
            order_planner->FinishInitialization();
            order_line_planner->FinishInitialization();
            item_planner->FinishInitialization();
            stock_planner->FinishInitialization();

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} initialization time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());

#if 0 // DEBUG
            {
                constexpr size_t max_print_size = 100u;
                constexpr size_t base_txn_size = TxnArray<TpccExecPlan>::kBaseTxnSize;
                uint32_t print_size = std::min(config.num_txns, max_print_size);
                uint32_t copy_size = print_size * base_txn_size;
                uint8_t execution_plan[max_print_size * base_txn_size];

                auto locToStr = [](uint32_t loc) -> std::string {
                    if (loc == loc_record_a)
                    {
                        return "RECORD_A";
                    }
                    else if (loc == loc_record_b)
                    {
                        return "RECORD_B";
                    }
                    else
                    {
                        return std::to_string(loc);
                    }
                };

                transferGpuToCpu(execution_plan, initialization_output.txns, copy_size);
                for (int i = 0; i < print_size; ++i)
                {
                    auto base_txn = reinterpret_cast<BaseTxn *>(execution_plan + i * base_txn_size);
                    if (1)
                    {
                        auto txn = reinterpret_cast<NewOrderExecPlan<FixedSizeTxn> *>(base_txn->data);
                        logger.Info("txn {} warehouse[{}] district[{}] customer[{}] new_order[{}] order[{}] "
                                    "item1[{}] stock_read1[{}] stock_write1[{}] order_line1[{}] "
                                    "item2[{}] stock_read2[{}] stock_write2[{}] order_line2[{}] "
                                    "item3[{}] stock_read3[{}] stock_write3[{}] order_line3[{}] "
                                    "item4[{}] stock_read4[{}] stock_write4[{}] order_line4[{}] "
                                    "item5[{}] stock_read5[{}] stock_write5[{}] order_line5[{}] ",
                            i, locToStr(txn->warehouse_loc), locToStr(txn->district_loc), locToStr(txn->customer_loc),
                            locToStr(txn->new_order_loc), locToStr(txn->order_loc),
                            locToStr(txn->item_plans[0].item_loc), locToStr(txn->item_plans[0].stock_read_loc),
                            locToStr(txn->item_plans[0].stock_write_loc), locToStr(txn->item_plans[0].orderline_loc),
                            locToStr(txn->item_plans[1].item_loc), locToStr(txn->item_plans[1].stock_read_loc),
                            locToStr(txn->item_plans[1].stock_write_loc), locToStr(txn->item_plans[1].orderline_loc),
                            locToStr(txn->item_plans[2].item_loc), locToStr(txn->item_plans[2].stock_read_loc),
                            locToStr(txn->item_plans[2].stock_write_loc), locToStr(txn->item_plans[2].orderline_loc),
                            locToStr(txn->item_plans[3].item_loc), locToStr(txn->item_plans[3].stock_read_loc),
                            locToStr(txn->item_plans[3].stock_write_loc), locToStr(txn->item_plans[3].orderline_loc),
                            locToStr(txn->item_plans[4].item_loc), locToStr(txn->item_plans[4].stock_read_loc),
                            locToStr(txn->item_plans[4].stock_write_loc), locToStr(txn->item_plans[4].orderline_loc));
                    }
                    if (0)
                    {
                        auto txn = reinterpret_cast<PaymentTxnExecPlan *>(base_txn->data);
                        logger.Info("txn {} warehouse[{}][{}] district[{}][{}] customer[{}][{}] history ", i,
                            locToStr(txn->warehouse_read_loc), locToStr(txn->warehouse_write_loc),
                            locToStr(txn->district_read_loc), locToStr(txn->district_write_loc),
                            locToStr(txn->customer_read_loc), locToStr(txn->customer_write_loc));
                    }
                }
            }
#endif
        }

        /* transfer */
        {
            start_time = std::chrono::high_resolution_clock::now();
            initialization_execution_plan_bridge.StartTransfer();
            index_execution_param_bridge.FinishTransfer();
            initialization_execution_plan_bridge.FinishTransfer();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} exec_transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* execution */
        {
#if 0 // DEBUG
            {
                constexpr size_t max_print_size = 100u;
                constexpr size_t base_txn_size = TxnArray<TpccTxnParam>::kBaseTxnSize;
                uint32_t print_size = std::min(config.num_txns, max_print_size);
                uint32_t copy_size = print_size * base_txn_size;
                uint8_t txn_params[max_print_size * base_txn_size];

                transferGpuToCpu(txn_params, initialization_input.txns, copy_size);
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
            start_time = std::chrono::high_resolution_clock::now();

            executor->execute(epoch_id);

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} execution time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }
    }
}
void TpccDb::indexEpoch(uint32_t epoch_id)
{
    /* TODO: remove */
    auto &logger = Logger::GetInstance();
    logger.Error("Deprecated function");
    exit(-1);

    //    /* zero-indexed */
    //    uint32_t index_epoch_id = epoch_id - 1;
    //
    //    /* it's important to index writes before reads */
    //    for (uint32_t i = 0; i < config.num_txns; ++i)
    //    {
    //        BaseTxn *txn = txn_array.getTxn(index_epoch_id, i);
    //        BaseTxn *txn_param = index_output.getTxn(i);
    //        index->indexTxnWrites(txn, txn_param, index_epoch_id);
    //    }
    //    for (uint32_t i = 0; i < config.num_txns; ++i)
    //    {
    //        BaseTxn *txn = txn_array.getTxn(index_epoch_id, i);
    //        BaseTxn *txn_param = index_output.getTxn(i);
    //        index->indexTxnReads(txn, txn_param, index_epoch_id);
    //    }
}
} // namespace epic::tpcc
