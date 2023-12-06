//
// Created by Shujian Qian on 2023-11-02.
//
#include <gacco/benchmarks/tpcc.h>

#include <memory>

#include <gacco/gpu_execution_planner.h>
#include <gacco/benchmarks/tpcc_txn_gen.h>
#include <util_log.h>
#include <util_gpu_transfer.h>
#include "gpu_allocator.h"
#include <gacco/benchmarks/tpcc_gpu_submitter.h>
#include <gacco/benchmarks/tpcc_gpu_executor.h>
#include <gacco/benchmarks/tpcc_storage.h>
#include <benchmarks/tpcc_gpu_index.h>

namespace gacco::tpcc {

using epic::BaseTxn;
using epic::tpcc::TpccGpuIndex;

TpccDb::TpccDb(TpccConfig config)
    : config(config)
    , txn_array(config.num_txns)
    , index_input(config.num_txns, config.index_device, false)
    , index_output(config.num_txns, config.index_device)
    , initialization_input(config.num_txns, config.initialize_device, false)
{
    for (int i = 0; i < config.epochs; ++i)
    {
        txn_array[i] = TxnArray<TpccTxn>(config.num_txns, epic::DeviceType::CPU);
    }
    //    index = std::make_shared<TpccCpuIndex>(config);
    index = std::make_shared<TpccGpuIndex>(config);
    input_index_bridge.Link(txn_array[0], index_input);
    index_initialization_bridge.Link(index_output, initialization_input);
    if (config.initialize_device == epic::DeviceType::GPU)
    {
        epic::GpuAllocator allocator;
        warehouse_planner = std::make_shared<GpuTableExecutionPlanner>(
            "warehouse", allocator, 0, 2, config.num_txns, config.warehouseTableSize());
        district_planner = std::make_shared<GpuTableExecutionPlanner>(
            "district", allocator, 0, 2, config.num_txns, config.districtTableSize());
        customer_planner = std::make_shared<GpuTableExecutionPlanner>(
            "customer", allocator, 0, 2, config.num_txns, config.customerTableSize());
        history_planner = std::make_shared<GpuTableExecutionPlanner>(
            "history", allocator, 0, 1, config.num_txns, config.historyTableSize());
        new_order_planner = std::make_shared<GpuTableExecutionPlanner>(
            "new_order", allocator, 0, 1, config.num_txns, config.newOrderTableSize());
        order_planner = std::make_shared<GpuTableExecutionPlanner>(
            "order", allocator, 0, 1, config.num_txns, config.orderTableSize());
        order_line_planner = std::make_shared<GpuTableExecutionPlanner>(
            "order_line", allocator, 0, 15, config.num_txns, config.orderLineTableSize());
        item_planner = std::make_shared<GpuTableExecutionPlanner>("item", allocator, 0, 15, config.num_txns, config.itemTableSize());
        stock_planner = std::make_shared<GpuTableExecutionPlanner>(
            "stock", allocator, 0, 15 * 2, config.num_txns, config.stockTableSize());

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
                stock_planner->curr_num_ops},
            config);
    }
    else
    {
        auto &logger = epic::Logger::GetInstance();
        logger.Error("Unsupported initialization device type");
        exit(-1);
    }

    if (config.execution_device == epic::DeviceType::GPU)
    {
        auto &logger = epic::Logger::GetInstance();
        logger.Info("Allocating records");
        TpccRecords records;
        epic::GpuAllocator allocator;
        records.warehouse_record =
            static_cast<WarehouseValue *>(allocator.Allocate(sizeof(WarehouseValue) * config.warehouseTableSize()));
        records.district_record =
            static_cast<DistrictValue *>(allocator.Allocate(sizeof(DistrictValue) * config.districtTableSize()));
        records.customer_record =
            static_cast<CustomerValue *>(allocator.Allocate(sizeof(CustomerValue) * config.customerTableSize()));
        /* TODO: history table is too big */
        //        records.history_record =
        //                static_cast<HistoryValue *>(allocator.Allocate(sizeof(HistoryValue) *
        //                config.historyTableSize()));
        records.new_order_record =
            static_cast<NewOrderValue *>(allocator.Allocate(sizeof(NewOrderValue) * config.newOrderTableSize()));
        records.order_record =
            static_cast<OrderValue *>(allocator.Allocate(sizeof(OrderValue) * config.orderTableSize()));
        records.order_line_record =
            static_cast<OrderLineValue *>(allocator.Allocate(sizeof(OrderLineValue) * config.orderLineTableSize()));
        records.item_record = static_cast<ItemValue *>(allocator.Allocate(sizeof(ItemValue) * config.itemTableSize()));
        records.stock_record =
            static_cast<StockValue *>(allocator.Allocate(sizeof(StockValue) * config.stockTableSize()));

        allocator.PrintMemoryInfo();

        executor = std::make_shared<GpuExecutor>(records,
            Executor::TpccTableLocks{warehouse_planner->table_lock, district_planner->table_lock,
                customer_planner->table_lock, history_planner->table_lock, new_order_planner->table_lock,
                order_planner->table_lock, order_line_planner->table_lock, item_planner->table_lock,
                stock_planner->table_lock},
            initialization_input, config);
    }
    else
    {
        auto &logger = epic::Logger::GetInstance();
        logger.Error("Unsupported initialize device");
        exit(-1);
    }
}

void TpccDb::loadInitialData()
{
    index->loadInitialData();
}

void TpccDb::generateTxns()
{
    auto &logger = epic::Logger::GetInstance();
    logger.Info("Generating {} txns with mix {} {} {} {} {}", config.num_txns * config.epochs, config.txn_mix.new_order,
        config.txn_mix.payment, config.txn_mix.order_status, config.txn_mix.delivery, config.txn_mix.stock_level);
    TpccTxnGenerator generator(config);
    for (size_t epoch = 0; epoch < config.epochs; ++epoch)
    {
        logger.Info("Generating epoch {}", epoch);
        for (size_t i = 0; i < config.num_txns; ++i)
        {
            BaseTxn *txn = txn_array[epoch].getTxn(i);
            uint32_t timestamp = epoch * config.num_txns + i;
            generator.generateTxn(txn, i, timestamp);
        }
    }
}

void TpccDb::indexEpoch(uint32_t epoch_id)
{
    /* TODO: remove */
    auto &logger = epic::Logger::GetInstance();
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

void TpccDb::runBenchmark()
{
    auto &logger = epic::Logger::GetInstance();
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

#if 0 // DEBUG
            {
                constexpr size_t max_print_size = 100u;
                constexpr size_t base_txn_size = TxnArray<TpccTxnParam>::kBaseTxnSize;
                uint32_t print_size = std::min(config.num_txns, max_print_size);
                for (int i = 0; i < print_size; ++i)
                {
                    uint8_t *txn_params = reinterpret_cast<uint8_t *>(index_output.txns);
                    auto param = &reinterpret_cast<TpccTxnParam *>(
                        reinterpret_cast<BaseTxn *>(txn_params + i * base_txn_size)->data)
                        ->new_order_txn;
                    uint32_t type = reinterpret_cast<BaseTxn *>(txn_params + i * base_txn_size)->txn_type;
                    logger.Info("txn {} type[{}] warehouse[{}] district[{}] customer[{}] order[{}] new_order[{}] numitems[{}] "
                                "item1[{}] stock_read1[{}] order_line1[{}] quantity1[{}] "
                                "item2[{}] stock_read2[{}] order_line2[{}] quantity2[{}] "
                                "item3[{}] stock_read3[{}] order_line3[{}] quantity3[{}] "
                                "item4[{}] stock_read4[{}] order_line4[{}] quantity4[{}] "
                                "item5[{}] stock_read5[{}] order_line5[{}] quantity5[{}] ",
                                i, type, param->warehouse_id, param->district_id, param->customer_id, param->order_id,
                                param->new_order_id, param->num_items, param->items[0].item_id, param->items[0].stock_id,
                                param->items[0].order_line_id, param->items[0].order_quantities, param->items[1].item_id,
                                param->items[1].stock_id, param->items[1].order_line_id, param->items[1].order_quantities,
                                param->items[2].item_id, param->items[2].stock_id, param->items[2].order_line_id,
                                param->items[2].order_quantities, param->items[3].item_id, param->items[3].stock_id,
                                param->items[3].order_line_id, param->items[3].order_quantities, param->items[4].item_id,
                                param->items[4].stock_id, param->items[4].order_line_id, param->items[4].order_quantities);
                }
            }
#endif
        }

        /* transfer */
        {
            start_time = std::chrono::high_resolution_clock::now();
            index_initialization_bridge.StartTransfer();
            index_initialization_bridge.FinishTransfer();
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

                epic::transferGpuToCpu(txn_params, initialization_input.txns, copy_size);
                for (int i = 0; i < print_size; ++i)
                {
                    auto param = &reinterpret_cast<TpccTxnParam *>(
                        reinterpret_cast<BaseTxn *>(txn_params + i * base_txn_size)->data)
                        ->new_order_txn;
                    uint32_t type = reinterpret_cast<BaseTxn *>(txn_params + i * base_txn_size)->txn_type;
                    logger.Info("txn {} type[{}] warehouse[{}] district[{}] customer[{}] order[{}] new_order[{}] numitems[{}] "
                                "item1[{}] stock_read1[{}] order_line1[{}] quantity1[{}] "
                                "item2[{}] stock_read2[{}] order_line2[{}] quantity2[{}] "
                                "item3[{}] stock_read3[{}] order_line3[{}] quantity3[{}] "
                                "item4[{}] stock_read4[{}] order_line4[{}] quantity4[{}] "
                                "item5[{}] stock_read5[{}] order_line5[{}] quantity5[{}] ",
                                i, type, param->warehouse_id, param->district_id, param->customer_id, param->order_id,
                                param->new_order_id, param->num_items, param->items[0].item_id, param->items[0].stock_id,
                                param->items[0].order_line_id, param->items[0].order_quantities, param->items[1].item_id,
                                param->items[1].stock_id, param->items[1].order_line_id, param->items[1].order_quantities,
                                param->items[2].item_id, param->items[2].stock_id, param->items[2].order_line_id,
                                param->items[2].order_quantities, param->items[3].item_id, param->items[3].stock_id,
                                param->items[3].order_line_id, param->items[3].order_quantities, param->items[4].item_id,
                                param->items[4].stock_id, param->items[4].order_line_id, param->items[4].order_quantities);
                }
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

        /* transfer */
        {
            start_time = std::chrono::high_resolution_clock::now();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} exec_transfer time: {} us", epoch_id,
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
                    if (0)
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
                    if (1)
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

        /* execution */
        {
            start_time = std::chrono::high_resolution_clock::now();

            executor->execute(epoch_id);

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} execution time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }
    }
}

} // namespace gacco::tpcc