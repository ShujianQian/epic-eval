//
// Created by Shujian Qian on 2024-01-25.
//

#ifndef EPIC_BENCHMARKS_YCSB_PVER_COPYER_H
#define EPIC_BENCHMARKS_YCSB_PVER_COPYER_H

#include <benchmarks/ycsb_storage.h>
#include <execution_planner.h>

namespace epic::ycsb {

void copyYcsbPver(
    YcsbRecords *records, YcsbVersions *versions, op_t *d_ops_to_copy, uint32_t *d_ver_to_copy, uint32_t num_copy);

}

#endif // EPIC_BENCHMARKS_YCSB_PVER_COPYER_H
