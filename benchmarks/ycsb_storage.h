//
// Created by Shujian Qian on 2023-11-22.
//

#ifndef EPIC_BENCHMARKS_YCSB_STORAGE_H
#define EPIC_BENCHMARKS_YCSB_STORAGE_H

#include <variant>
#include <storage.h>
#include <benchmarks/ycsb_table.h>

namespace epic::ycsb {

using YcsbVersions = Version<YcsbValue>;
using YcsbRecords = Record<YcsbValue>;

using YcsbFieldVersions = Version<YcsbFieldValue>;
using YcsbFieldRecords = Record<YcsbFieldValue>;

using YcsbVersionArrType = std::variant<YcsbVersions *, YcsbFieldVersions *>;
using YcsbRecordArrType = std::variant<YcsbRecords *, YcsbFieldRecords *>;

}

#endif // EPIC_BENCHMARKS_YCSB_STORAGE_H
