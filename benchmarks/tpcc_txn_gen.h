//
// Created by Shujian Qian on 2023-11-05.
//

#ifndef EPIC_BENCHMARKS_TPCC_TXN_GEN_H
#define EPIC_BENCHMARKS_TPCC_TXN_GEN_H

#include <cstdint>
#include <random>
#include <cassert>

#include <tpcc_config.h>
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
public:
    inline uint32_t getOId(uint32_t timestamp)
    {
        constexpr uint32_t loader_max_order_id = 1'000'000; /* max number of orders used by loader */
        return timestamp + loader_max_order_id;
    }
};

struct TpccTxnGenerator
{
    std::mt19937_64 gen;
    std::uniform_int_distribution<uint32_t> txn_type_dist, w_id_dist, d_id_dist, num_item_dist, percentage_gen,
        order_quantity_dist, payment_amount_dist;
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
        , config(config)
    {}

    TpccTxnType getTxnType();
    void generateTxn(NewOrderTxnInput<FixedSizeTxn> *txn, uint32_t timestamp);
    void generateTxn(PaymentTxnInput *txn, uint32_t timestamp);
    void generateTxn(OrderStatusTxn *txn, uint32_t timestamp);
    void generateTxn(DeliveryTxn *txn, uint32_t timestamp);
    void generateTxn(StockLevelTxn *txn, uint32_t timestamp);
    void generateTxn(BaseTxn *txn, uint32_t timestamp);
};

}

#endif // EPIC_BENCHMARKS_TPCC_TXN_GEN_H
