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
    , index(this->config)
    , txn_array(config.num_txns, config.epochs)
    , index_output(config.num_txns, config.index_device)
    , initialization_input(config.num_txns, config.initialize_device, false)
    , initialization_output(config.num_txns, config.initialize_device)
    , loader(this->config, index)
{
    index_initialization_bridge.Link(index_output, initialization_input);
    if (config.initialize_device == DeviceType::GPU)
    {
        GpuAllocator allocator;
        warehouse_planner = std::make_unique<GpuTableExecutionPlanner>(
            "warehouse", allocator, 0, 1, config.num_txns, config.num_warehouses, initialization_output);
        district_planner = std::make_unique<GpuTableExecutionPlanner>(
            "district", allocator, 0, 1, config.num_txns, config.num_warehouses * 10, initialization_output);
        customer_planner = std::make_unique<GpuTableExecutionPlanner>(
            "customer", allocator, 0, 1, config.num_txns, config.num_warehouses * 10 * 3000, initialization_output);
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
}

namespace {

class TpccNuRand
{
public:
    uint32_t C, x, y;
    std::uniform_int_distribution<uint32_t> dist1, dist2;
    TpccNuRand(uint32_t A, uint32_t x, uint32_t y)
        : C(std::random_device{}())
        , x(x)
        , y(y)
        , dist1(0, A)
        , dist2(x, y)
    {
        assert((x == 0 && y == 999 && A == 255) || (x == 1 && y == 3000 && A == 1023) ||
               (x == 1 && y == 100000 && A == 8191));
    }

    template<typename RandomEngine>
    uint32_t operator()(RandomEngine &gen)
    {
        uint32_t rand = (dist1(gen) | dist2(gen)) + C;
        return (rand % (y - x + 1)) + x;
    }
};

class TpccOIdGenerator
{
public:
    uint32_t getOId(uint32_t timestamp)
    {
        return timestamp;
    }
};

struct TpccTxnGenerator
{
    std::mt19937_64 gen;
    std::uniform_int_distribution<uint32_t> txn_type_dist, w_id_dist, d_id_dist, num_item_dist, remote_order_dist,
        order_quantity_dist;
    TpccNuRand c_id_dist, i_id_dist;
    TpccOIdGenerator o_id_gen;

    const TpccConfig &config;

    TpccTxnGenerator(const TpccConfig &config)
        : gen(std::random_device{}())
        , txn_type_dist(0, 99)
        , w_id_dist(1, config.num_warehouses)
        , d_id_dist(1, 10)
        , c_id_dist(1023, 1, 3000)
        , i_id_dist(8191, 1, 100'000)
        , num_item_dist(5, 15)
        , remote_order_dist(1, 100)
        , order_quantity_dist(1, 10)
        , config(config)
    {}

    TpccTxnType getTxnType()
    {
        uint32_t txn_type = txn_type_dist(gen);
        uint32_t acc = 0;
        if (txn_type < acc + config.txn_mix.new_order)
        {
            return TpccTxnType::NEW_ORDER;
        }
        acc += config.txn_mix.new_order;
        if (txn_type < acc + config.txn_mix.payment)
        {
            return TpccTxnType::PAYMENT;
        }
        acc += config.txn_mix.payment;
        if (txn_type < acc + config.txn_mix.order_status)
        {
            return TpccTxnType::ORDER_STATUS;
        }
        acc += config.txn_mix.order_status;
        if (txn_type < acc + config.txn_mix.delivery)
        {
            return TpccTxnType::DELIVERY;
        }
        return TpccTxnType::STOCK_LEVEL;
    }

    void generateTxn(NewOrderTxnInput<FixedSizeTxn> *txn, uint32_t timestamp)
    {
        /* TODO: generate rollbacks? */
        txn->origin_w_id = w_id_dist(gen);
        txn->o_id = o_id_gen.getOId(timestamp);
        txn->d_id = d_id_dist(gen);
        txn->c_id = c_id_dist(gen);
        txn->num_items = num_item_dist(gen);
        for (size_t i = 0; i < txn->num_items; ++i)
        {
            /* generate unique item indexes */
            bool retry;
            do
            {
                txn->items[i].i_id = i_id_dist(gen);
                retry = false;
                for (size_t j = 0; j + 1 < i; ++j)
                {
                    if (txn->items[i].i_id == txn->items[j].i_id)
                    {
                        retry = true;
                        break;
                    }
                }
            } while (retry);

            /* no remote warehouse for single warehouse configs, otherwise 1% remote orders */
            bool supply_from_remote = config.num_warehouses > 1 && (remote_order_dist(gen) == 1);
            if (supply_from_remote)
            {
                do
                {
                    txn->items[i].w_id = w_id_dist(gen);
                } while (txn->items[i].w_id == txn->origin_w_id);
            }
            else
            {
                txn->items[i].w_id = txn->origin_w_id;
            }

            txn->items[i].order_quantities = order_quantity_dist(gen);
        }
    }

    void generateTxn(PaymentTxn *txn, uint32_t timestamp)
    {
        /* TODO: generate payment txn */
    }

    void generateTxn(OrderStatusTxn *txn, uint32_t timestamp)
    {
        /* TODO: generate order-status txn */
    }

    void generateTxn(DeliveryTxn *txn, uint32_t timestamp)
    {
        /* TODO: generate delivery txn */
    }

    void generateTxn(StockLevelTxn *txn, uint32_t timestamp)
    {
        /* TODO: generate stock-level txn */
    }

    void generateTxn(BaseTxn *txn, uint32_t timestamp)
    {
        TpccTxnType txn_type = getTxnType();
        txn->txn_type = static_cast<uint32_t>(txn_type);
        switch (txn_type)
        {
        case TpccTxnType::NEW_ORDER:
            generateTxn(reinterpret_cast<NewOrderTxnInput<FixedSizeTxn> *>(txn->data), timestamp);
            break;
        case TpccTxnType::PAYMENT:
            generateTxn(reinterpret_cast<PaymentTxn *>(txn->data), timestamp);
            break;
        case TpccTxnType::ORDER_STATUS:
            generateTxn(reinterpret_cast<OrderStatusTxn *>(txn->data), timestamp);
            break;
        case TpccTxnType::DELIVERY:
            generateTxn(reinterpret_cast<DeliveryTxn *>(txn->data), timestamp);
            break;
        case TpccTxnType::STOCK_LEVEL:
            generateTxn(reinterpret_cast<StockLevelTxn *>(txn->data), timestamp);
            break;
        default:
            break;
        }
    }
};
} // namespace

void TpccDb::generateTxns()
{
    auto &logger = Logger::GetInstance();

    TpccTxnGenerator generator(config);
    for (size_t epoch = 0; epoch < config.epochs; ++epoch)
    {
        logger.Info("Generating epoch {}", epoch);
        for (size_t i = 0; i < config.num_txns; ++i)
        {
            BaseTxn *txn = txn_array.getTxn(epoch, i);
            uint32_t timestamp = epoch * config.num_txns + i;
            generator.generateTxn(txn, timestamp);
        }
    }
}

void TpccDb::loadInitialData()
{
    loader.loadInitialData();
}

void TpccDb::runBenchmark()
{
    auto &logger = Logger::GetInstance();
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    for (uint32_t epoch_id = 0; epoch_id < config.epochs; ++epoch_id)
    {
        logger.Info("Running epoch {}", epoch_id);
        /* index */
        {
            start_time = std::chrono::high_resolution_clock::now();
            indexEpoch(epoch_id);
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} indexing time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }

        /* transfer */
        {
            start_time = std::chrono::high_resolution_clock::now();
            index_initialization_bridge.StartTransfer();
            index_initialization_bridge.FinishTransfer();
            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} transfer time: {} us", epoch_id,
                std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
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
                    if (loc == loc_record_a) {
                        return "RECORD_A";
                    } else if (loc == loc_record_b) {
                        return "RECORD_B";
                    } else {
                        return std::to_string(loc);
                    }
                };

                transferGpuToCpu(execution_plan, initialization_output.txns, copy_size);
                for (int i = 0; i < print_size; ++i)
                {
                    auto base_txn = reinterpret_cast<BaseTxn *>(execution_plan + i * base_txn_size);
                    auto txn = reinterpret_cast<NewOrderExecPlan<FixedSizeTxn> *>(base_txn->data);
                    logger.Info("txn {} warehouse[{}] district[{}] customer[{}] new_order[{}] order[{}] "
                                "item1[{}] stock_read1[{}] stock_write1[{}] order_line1[{}] "
                                "item2[{}] stock_read2[{}] stock_write2[{}] order_line2[{}] "
                                "item3[{}] stock_read3[{}] stock_write3[{}] order_line3[{}] "
                                "item4[{}] stock_read4[{}] stock_write4[{}] order_line4[{}] "
                                "item5[{}] stock_read5[{}] stock_write5[{}] order_line5[{}] ",
                        i, locToStr(txn->warehouse_loc), locToStr(txn->district_loc), locToStr(txn->customer_loc), locToStr(txn->new_order_loc), locToStr(txn->order_loc),
                        locToStr(txn->item_plans[0].item_loc), locToStr(txn->item_plans[0].stock_read_loc), locToStr(txn->item_plans[0].stock_write_loc), locToStr(txn->item_plans[0].orderline_loc),
                        locToStr(txn->item_plans[1].item_loc), locToStr(txn->item_plans[1].stock_read_loc), locToStr(txn->item_plans[1].stock_write_loc), locToStr(txn->item_plans[1].orderline_loc),
                        locToStr(txn->item_plans[2].item_loc), locToStr(txn->item_plans[2].stock_read_loc), locToStr(txn->item_plans[2].stock_write_loc), locToStr(txn->item_plans[2].orderline_loc),
                        locToStr(txn->item_plans[3].item_loc), locToStr(txn->item_plans[3].stock_read_loc), locToStr(txn->item_plans[3].stock_write_loc), locToStr(txn->item_plans[3].orderline_loc),
                        locToStr(txn->item_plans[4].item_loc), locToStr(txn->item_plans[4].stock_read_loc), locToStr(txn->item_plans[4].stock_write_loc), locToStr(txn->item_plans[4].orderline_loc));
                }
            }
#endif
        }

        /* execution */
        {
            start_time = std::chrono::high_resolution_clock::now();

            end_time = std::chrono::high_resolution_clock::now();
            logger.Info("Epoch {} execution time: {} us", epoch_id,
                        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
        }
    }
}
void TpccDb::indexEpoch(uint32_t epoch_id)
{
    /* it's important to index writes before reads */
    for (uint32_t i = 0; i < config.num_txns; ++i)
    {
        BaseTxn *txn = txn_array.getTxn(epoch_id, i);
        BaseTxn *txn_param = index_output.getTxn(i);
        index.indexTxnWrites(txn, txn_param, epoch_id);
    }
    for (uint32_t i = 0; i < config.num_txns; ++i)
    {
        BaseTxn *txn = txn_array.getTxn(epoch_id, i);
        BaseTxn *txn_param = index_output.getTxn(i);
        index.indexTxnReads(txn, txn_param, epoch_id);
    }
}
} // namespace epic::tpcc
