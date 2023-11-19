//
// Created by Shujian Qian on 2023-11-08.
//

#ifndef EPIC__UTIL_CUB_TEMP_ARR_SIZE_CUH
#define EPIC__UTIL_CUB_TEMP_ARR_SIZE_CUH

#include <cub/cub.cuh>
#include <algorithm>

namespace epic {

template<typename InputT>
size_t getCubTempArraySize(size_t num_items, size_t max_num_rows)
{
    void *d_temp_storage = nullptr;
    size_t max_storage_bytes = 0;
    size_t temp_storage_bytes = 0;
    void *dummy_pointer = nullptr;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, reinterpret_cast<InputT *>(dummy_pointer),
        reinterpret_cast<InputT *>(dummy_pointer), num_items);
    max_storage_bytes = std::max(max_storage_bytes, temp_storage_bytes);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, reinterpret_cast<InputT *>(dummy_pointer),
        reinterpret_cast<InputT *>(dummy_pointer), num_items);
    max_storage_bytes = std::max(max_storage_bytes, temp_storage_bytes);
    cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, reinterpret_cast<InputT *>(dummy_pointer),
        reinterpret_cast<InputT *>(dummy_pointer), reinterpret_cast<InputT *>(dummy_pointer),
        reinterpret_cast<InputT *>(dummy_pointer), max_num_rows);
    max_storage_bytes = std::max(max_storage_bytes, temp_storage_bytes);
    return max_storage_bytes;
}

} // namespace epic

#endif // EPIC__UTIL_CUB_TEMP_ARR_SIZE_CUH
