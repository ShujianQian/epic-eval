//
// Created by Shujian Qian on 2023-11-05.
//

#ifndef EPIC_BENCHMARKS_TPCC_TXN_GEN_H
#define EPIC_BENCHMARKS_TPCC_TXN_GEN_H

#include <cstdint>
#include <random>
#include <cassert>

#include "tpcc_config.h"
#include <benchmarks/tpcc_txn.h>

namespace epic::tpcc {

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
    inline uint32_t operator()(RandomEngine &gen)
    {
        uint32_t rand = (dist1(gen) | dist2(gen)) + C;
        return (rand % (y - x + 1)) + x;
    }
};

class TpccOIdGenerator
{
    uint32_t next_oid_counter[256][20]{};
    uint32_t next_deliver_oid_counter[256]{};
public:
    explicit TpccOIdGenerator(uint32_t initial_orders_per_district = 3000, uint32_t initial_delivered_orders_per_district = 2100)
    {
        for (size_t i = 0; i < 256; ++i)
        {
            for (size_t j = 0; j < 20; ++j)
            {
                next_oid_counter[i][j] = initial_orders_per_district;
            }
            next_deliver_oid_counter[i] = initial_delivered_orders_per_district;
        }
    }
    inline uint32_t getOId(uint32_t timestamp)
    {
        constexpr uint32_t loader_max_order_id = 1'000'000; /* max number of orders used by loader */
        return timestamp + loader_max_order_id;
    }

    inline uint32_t nextOId(uint32_t w_id, uint32_t d_id) {
        return ++next_oid_counter[w_id][d_id];
    }

    inline uint32_t currOId(uint32_t w_id, uint32_t d_id) {
        return next_oid_counter[w_id][d_id];
    }

    inline uint32_t nextDeliverOId(const uint32_t w_id)
    {
        uint32_t retval = ++next_deliver_oid_counter[w_id];
        for (int d_id = 0; d_id < 10; ++d_id)
        {
            assert(retval < next_oid_counter[w_id][d_id]);
        }
        return retval;
    }
};

struct TpccTxnGenerator
{
    std::mt19937_64 gen;
    std::uniform_int_distribution<uint32_t> txn_type_dist, w_id_dist, d_id_dist, num_item_dist, percentage_gen,
        order_quantity_dist, payment_amount_dist, carrier_id_dist;
    TpccNuRand c_id_dist, i_id_dist;
    TpccOIdGenerator o_id_gen;

    const TpccConfig config;

    TpccTxnGenerator(const TpccConfig &config)
        : gen(std::random_device{}())
        , txn_type_dist(0, 99)
        , w_id_dist(1, config.num_warehouses)
        , d_id_dist(1, 10)
        , c_id_dist(1023, 1, 3000)
        , i_id_dist(8191, 1, 100'000)
        , num_item_dist(5, 15)
        , percentage_gen(1, 100)
        , order_quantity_dist(1, 10)
        , payment_amount_dist(1, 5000)
        , carrier_id_dist(1, 10)
        , config(config)
    {}

    TpccTxnType getTxnType();
    void generateTxn(NewOrderTxnInput<FixedSizeTxn> *txn, uint32_t timestamp);
    void generateTxn(PaymentTxnInput *txn, uint32_t timestamp);
    void generateTxn(OrderStatusTxnInput *txn, uint32_t timestamp);
    void generateTxn(DeliveryTxnInput *txn, uint32_t timestamp);
    void generateTxn(StockLevelTxnInput *txn, uint32_t timestamp);
    void generateTxn(BaseTxn *txn, uint32_t timestamp);
};

}

#endif // EPIC_BENCHMARKS_TPCC_TXN_GEN_H
