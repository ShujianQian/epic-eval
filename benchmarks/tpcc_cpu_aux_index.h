//
// Created by Shujian Qian on 2024-04-04.
//

#ifndef TPCC_CPU_AUX_INDEX_H
#define TPCC_CPU_AUX_INDEX_H

#include <cstdint>

#include <cpu_auxiliary_range_index.h>
#include <benchmarks/tpcc_table.h>
#include <benchmarks/tpcc_txn_gen.h>
#include <txn.h>
#include <util_math.h>

namespace epic::tpcc {

class TpccCpuAuxIndex
{
private:
    CpuAuxRangeIndex<ClientOrderKey::baseType, uint32_t> customer_order_index;
    uint32_t num_slots_per_district;
    std::vector<uint32_t> order_num_items, order_customers;
    std::vector<uint32_t[15]> order_items;
    TpccConfig config;
    TpccNuRand i_id_dist;

public:
    explicit TpccCpuAuxIndex(TpccConfig &config)
        : config(config)
        , num_slots_per_district(config.num_txns / config.num_warehouses / 10 * config.epochs)
        , order_num_items(config.num_warehouses * 10 * num_slots_per_district, 0)
        , order_customers(config.num_warehouses * 10 * num_slots_per_district, 0)
        , order_items(config.num_warehouses * 10 * num_slots_per_district)
        , i_id_dist(8191, 1, 100'000)
    {
        auto &logger = Logger::GetInstance();
        logger.Info("Num slots per district: {}", num_slots_per_district);
        logger.Info("CPU Aux index order_num_items memory size: {}",
            formatSizeBytes(order_num_items.capacity() * sizeof(decltype(order_num_items)::value_type)));
        logger.Info("CPU Aux index order_customers memory size: {}",
            formatSizeBytes(order_customers.capacity() * sizeof(decltype(order_customers)::value_type)));
        logger.Info("CPU Aux index order_items memory size: {}",
            formatSizeBytes(order_items.capacity() * sizeof(decltype(order_items)::value_type)));
    }

    void loadInitialData()
    {
        std::mt19937_64 gen(std::random_device{}());
        for (uint32_t w_id = 1; w_id <= config.num_warehouses; w_id++)
        {
            for (uint32_t d_id = 1; d_id <= 10; d_id++)
            {
                for (uint32_t o_id = 1; o_id <= 3000; o_id++)
                {
                    const uint32_t c_id = o_id; // assume one order per customer
                    const uint32_t sid = 0; // initial sid
                    const ClientOrderKey key{o_id, c_id, d_id, w_id};
                    customer_order_index.searchOrInsert(key.base_key, sid);

                    const uint32_t o_num_items = 15;
                    const uint32_t district_id = d_id - 1+ (w_id - 1) * 10;
                    order_num_items[district_id * num_slots_per_district + o_id] = o_num_items; // TODO: randomize num_items for each order
                    order_customers[district_id * num_slots_per_district + o_id] = c_id;

                    uint32_t(&o_items)[15] = order_items[district_id * num_slots_per_district + o_id];
                    for (uint32_t ol_id = 0; ol_id < o_num_items; ++ol_id)
                    {
                        o_items[ol_id] = i_id_dist(gen);
                    }
                }
            }
        }
    }

    void insertTxnUpdates(TxnArray<TpccTxn> &txns, size_t epoch)
    {
        // for (int i = 0; i < txns.num_txns; i++)
        // {
        //   BaseTxn *base_txn = txns.getTxn(i);
        //   TpccTxn *txn = reinterpret_cast<TpccTxn *>(base_txn->data);
        //   switch (static_cast<TpccTxnType>(base_txn->txn_type))
        //   {
        //   case TpccTxnType::NEW_ORDER:
        //   {
        //     NewOrderKey key(txn->new_order_txn.o_id, txn->new_order_txn.d_id, txn->new_order_txn.origin_w_id);
        //     new_order_index.searchOrInsert(key.base_key, i);
        //     break;
        //   }
        //   default:
        //     break;
        //   }
        // }

        std::atomic<uint32_t> txn_id = 0;
        std::vector<std::thread> threads;
        threads.reserve(config.cpu_exec_num_threads);
        for (int thread_id = 0; thread_id < config.cpu_exec_num_threads; thread_id++)
        {
            threads.emplace_back([this, &txns, epoch, &txn_id]() {
                constexpr uint32_t batch_size = 512;
                uint32_t local_txn_id = txn_id.fetch_add(batch_size);
                while (local_txn_id < txns.num_txns)
                {
                    uint32_t end = std::min(local_txn_id + batch_size, static_cast<uint32_t>(txns.num_txns));
                    for (uint32_t txn_id = local_txn_id; txn_id < end; ++txn_id)
                    {
                        BaseTxn *base_txn = txns.getTxn(txn_id);
                        TpccTxn *txn = reinterpret_cast<TpccTxn *>(base_txn->data);
                        switch (static_cast<TpccTxnType>(base_txn->txn_type))
                        {
                        case TpccTxnType::NEW_ORDER: {
                            ClientOrderKey key{txn->new_order_txn.o_id, txn->new_order_txn.c_id,
                                txn->new_order_txn.d_id, txn->new_order_txn.origin_w_id};
                            const uint32_t sid = epoch << 24 | txn_id;
                            customer_order_index.searchOrInsert(key.base_key, sid);

                            uint32_t district_id = txn->new_order_txn.d_id - 1 + (txn->new_order_txn.origin_w_id - 1)* 10;
                            order_num_items[district_id * num_slots_per_district + txn->new_order_txn.o_id] =
                                txn->new_order_txn.num_items;
                            order_customers[district_id * num_slots_per_district + txn->new_order_txn.o_id] =
                                txn->new_order_txn.c_id;

                            uint32_t(&o_items)[15] = order_items[district_id * num_slots_per_district + txn->new_order_txn.o_id];
                            for (uint32_t ol_id = 0; ol_id < txn->new_order_txn.num_items; ++ol_id)
                            {
                                o_items[ol_id] = txn->new_order_txn.items[ol_id].i_id;
                            }
                            break;
                        }
                        case TpccTxnType::PAYMENT:
                            case TpccTxnType::STOCK_LEVEL:
                            case TpccTxnType::DELIVERY:
                        case TpccTxnType::ORDER_STATUS:
                            break;
                        default:
                            assert(false);
                            break;
                        }
                    }
                    local_txn_id = txn_id.fetch_add(batch_size);
                }
            });
        }
        for (auto &t : threads)
        {
            t.join();
        }
    }

    void performRangeQueries(TxnArray<TpccTxn> &txns, size_t epoch)
    {
        std::atomic<uint32_t> txn_id = 0;
        std::vector<std::thread> threads;
        threads.reserve(config.cpu_exec_num_threads);
        for (int thread_id = 0; thread_id < config.cpu_exec_num_threads; thread_id++)
        {
            threads.emplace_back([this, &txns, epoch, &txn_id]() {
                constexpr uint32_t batch_size = 512;
                uint32_t local_txn_id = txn_id.fetch_add(batch_size);
                while (local_txn_id < txns.num_txns)
                {
                    uint32_t end = std::min(local_txn_id + batch_size, static_cast<uint32_t>(txns.num_txns));
                    for (uint32_t txn_id = local_txn_id; txn_id < end; ++txn_id)
                    {
                        BaseTxn *base_txn = txns.getTxn(txn_id);
                        TpccTxn *txn = reinterpret_cast<TpccTxn *>(base_txn->data);
                        switch (static_cast<TpccTxnType>(base_txn->txn_type))
                        {
                        case TpccTxnType::ORDER_STATUS: {
                            ClientOrderKey key{txn->order_status_txn.o_id, txn->order_status_txn.c_id,
                                txn->order_status_txn.d_id, txn->order_status_txn.w_id};
                            auto it = customer_order_index.searchReverseIterator(key.base_key, 0);
                            if (!it->is_valid())
                            {
                                auto &logger = epic::Logger::GetInstance();
                                logger.Error("ORDER_STATUS: key not found c_id[{}] d_id[{}] w_id[{}]",
                                    txn->order_status_txn.c_id, txn->order_status_txn.d_id, txn->order_status_txn.w_id);
                                break;
                            }
                            key.base_key = it->getKey();
                            txn->order_status_txn.o_id = key.oc_o_id;
                            const uint32_t district_id =
                                txn->order_status_txn.d_id - 1 + (txn->order_status_txn.w_id - 1) * 10;
                            txn->order_status_txn.num_items =
                                order_num_items[district_id * num_slots_per_district + txn->order_status_txn.o_id];
                            break;
                        }
                        case TpccTxnType::DELIVERY: {
                            for (uint32_t d_id = 1; d_id <= 10; ++d_id)
                            {
                                uint32_t district_id = d_id - 1 + (txn->delivery_txn.w_id - 1) * 10;
                                txn->delivery_txn.num_items[d_id - 1] =
                                    order_num_items[district_id * num_slots_per_district + txn->delivery_txn.o_id];
                                txn->delivery_txn.customers[d_id - 1] =
                                    order_customers[district_id * num_slots_per_district + txn->delivery_txn.o_id];
                            }
                            // uint32_t num_items = 0;
                            // for (int i = 0; i < 10; ++i)
                            // {
                            //     num_items += txn->delivery_txn.num_items[i];
                            // }
                            // auto &logger = epic::Logger::GetInstance();
                            // logger.Info("DELIVERY: w_id[{}] o_id[{}] num_items[{}]", txn->delivery_txn.w_id,
                            // txn->delivery_txn.o_id, num_items);
                            break;
                        }
                        case TpccTxnType::STOCK_LEVEL: {
                            uint32_t num_items = 0;
                            for (uint32_t o_id = txn->stock_level_txn.o_id; o_id > txn->stock_level_txn.o_id - 20;
                                 --o_id)
                            {
                                uint32_t district_id =
                                    txn->stock_level_txn.d_id - 1 + (txn->stock_level_txn.w_id - 1) * 10;
                                uint32_t order_idx = district_id * num_slots_per_district + o_id;
                                memcpy(&(txn->stock_level_txn.items[num_items]), order_items[order_idx],
                                    sizeof(uint32_t) * order_num_items[order_idx]);
                                num_items += order_num_items[order_idx];
                            }
                            std::sort(txn->stock_level_txn.items, &(txn->stock_level_txn.items[num_items]));
                            txn->stock_level_txn.num_items = num_items;
                            break;
                        }
                        case TpccTxnType::PAYMENT:
                        case TpccTxnType::NEW_ORDER:
                            break;
                        default:
                            assert(false);
                            break;
                        }
                    }
                    local_txn_id = txn_id.fetch_add(batch_size);
                }
            });
        }
        for (auto &t : threads)
        {
            t.join();
        }
    }
};

} // namespace epic::tpcc

#endif // TPCC_CPU_AUX_INDEX_H
