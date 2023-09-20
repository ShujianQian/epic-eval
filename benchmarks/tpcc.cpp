//
// Created by Shujian Qian on 2023-09-15.
//

#include "benchmarks/tpcc.h"

#include <cassert>
#include <random>
#include <cstdio>

#include "benchmarks/tpcc_txn.h"

#include "util_log.h"

namespace epic::tpcc {
TpccTxnMix::TpccTxnMix(uint32_t new_order, uint32_t payment, uint32_t order_status, uint32_t delivery, uint32_t stock_level)
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
    , loader(this->config, index)
{}

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
        assert((x == 0 && y == 999 && A == 255) || (x == 1 && y == 3000 && A == 1023) || (x == 1 && y == 100000 && A == 8191));
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
    std::uniform_int_distribution<uint32_t> txn_type_dist, w_id_dist, d_id_dist, num_item_dist, remote_order_dist, order_quantity_dist;
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
} // namespace epic::tpcc
