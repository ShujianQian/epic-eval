//
// Created by Shujian Qian on 2023-08-15.
//

#ifndef TPCC_H
#define TPCC_H

#include "tpcc_config.h"
#include "tpcc_common.h"
#include "tpcc_table.h"
#include "tpcc_txn.h"

#include <memory>

#include "execution_planner.h"
#include "table.h"

namespace epic::tpcc {

class TpccDb
{
private:
    ExecutionPlanner *planner;

    Table warehouse_table;

    TpccConfig config;
    TxnArray<TpccTxn> txn_array;

    TpccIndex index;
    TpccLoader loader;

public:
    TpccDb(TpccConfig config);

    void loadInitialData();
    void generateTxns();

private:
};

} // namespace epic::tpcc

#endif // TPCC_H
