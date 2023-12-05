//
// Created by Shujian Qian on 2023-10-25.
//

#ifndef STORAGE_H
#define STORAGE_H

#include <cstdint>
#include <xmmintrin.h>

#include <execution_planner.h>
#include <util_opt.h>
#include <util_arch.h>

namespace epic {

template<typename ValueType>
struct Record
{
    uint32_t version1 = 0, version2 = 0;
    ValueType value1, value2;
} __attribute__((aligned(kDeviceCacheLineSize)));

/* make sure Version is properly aligned for GPU atomic operations */
static_assert(sizeof(Record<int>) == kDeviceCacheLineSize);

/* make sure that the two versions are adjacent in memory, so they can be atomically read with 64bit instr */
static_assert(offsetof(Record<int>, version2) - offsetof(Record<int>, version1) == sizeof(uint32_t));

template<typename ValueType>
struct Version
{
    uint32_t version = 0;
    ValueType value;
} __attribute__((aligned(kDeviceCacheLineSize)));

/* make sure Version is properly aligned for GPU atomic operations */
static_assert(sizeof(Version<int>) == kDeviceCacheLineSize);

template<typename ValueType>
EPIC_FORCE_INLINE void readFromTable(Record<ValueType> *record, Version<ValueType> *version, uint32_t record_id,
    uint32_t read_loc, uint32_t epoch, ValueType *result)
{
    /* record a read */
    if (read_loc == loc_record_a)
    {
        /* reading the version from previous epoch, no syncronization needed */
        uint64_t combined_versions =
            __atomic_load_n(reinterpret_cast<uint64_t *>(&record[record_id].version1), __ATOMIC_SEQ_CST);
        uint32_t version1 = combined_versions & 0xFFFFFFFF;
        uint32_t version2 = combined_versions >> 32;
        ValueType *value_to_read = nullptr;
        if (version1 == epoch)
        {
            /* version1 is written in this epoch (record_b) */
            value_to_read = &record[record_id].value2;
        }
        else if (version2 == epoch)
        {
            /* version1 is written in this epoch (record_b) */
            value_to_read = &record[record_id].value1;
        }
        else if (version1 < version2)
        {
            /* version2 is the latest version before this epoch (record_a) */
            value_to_read = &record[record_id].value2;
        }
        else
        {
            /* version1 is the latest version before this epoch (record_a) */
            value_to_read = &record[record_id].value1;
        }
        memcpy(result, value_to_read, sizeof(ValueType));
        return;
    }

    /* record b read */
    if (read_loc == loc_record_b)
    {
        uint64_t combined_versions =
            __atomic_load_n(reinterpret_cast<uint64_t *>(&record[record_id].version1), __ATOMIC_SEQ_CST);
        uint32_t version1 = combined_versions & 0xFFFFFFFF;
        uint32_t version2 = combined_versions >> 32;
        ValueType *value_to_read = nullptr;
        if (version1 == epoch)
        {
            /* version1 is written in this epoch (record_b) */
            value_to_read = &record[record_id].value1;
        }
        else if (version2 == epoch)
        {
            /* version1 is written in this epoch (record_b) */
            value_to_read = &record[record_id].value2;
        }
        else if (version1 < version2)
        {
            /* version1 will be written in this epoch (record_b) */
            value_to_read = &record[record_id].value1;
            while (__atomic_load_n(&record[record_id].version1, __ATOMIC_SEQ_CST) != epoch)
            {
                _mm_pause();
            }
        }
        else
        {
            /* version2 will be written in this epoch (record_b) */
            value_to_read = &record[record_id].value2;
            while (__atomic_load_n(&record[record_id].version2, __ATOMIC_SEQ_CST) != epoch)
            {
                _mm_pause();
            }
        }
        memcpy(result, value_to_read, sizeof(ValueType));
        return;
    }

    /* version read */
    while (__atomic_load_n(&version[read_loc].version, __ATOMIC_SEQ_CST) != epoch)
    {
        _mm_pause();
    }
    memcpy(result, &version[read_loc].value, sizeof(ValueType));
}

template<typename ValueType>
EPIC_FORCE_INLINE void writeToTable(Record<ValueType> *record, Version<ValueType> *version, uint32_t record_id,
    uint32_t write_loc, uint32_t epoch, ValueType *source)
{
    if (write_loc == loc_record_b)
    {
        uint64_t combined_versions =
            __atomic_load_n(reinterpret_cast<uint64_t *>(&record[record_id].version1), __ATOMIC_SEQ_CST);
        /* TODO: I don't think atomic read is required here */
        uint32_t version1 = combined_versions & 0xFFFFFFFF;
        uint32_t version2 = combined_versions >> 32;
        if (version1 < version2)
        {
            /* version2 is the latest version before this epoch (record_a) */
            memcpy(&record[record_id].value1, source, sizeof(ValueType));
            __atomic_store_n(&record[record_id].version1, epoch, __ATOMIC_SEQ_CST);
        }
        else
        {
            /* version1 is the latest version before this epoch (record_a) */
            memcpy(&record[record_id].value2, source, sizeof(ValueType));
            __atomic_store_n(&record[record_id].version2, epoch, __ATOMIC_SEQ_CST);
        }
        return;
    }

    /* version write */
//    memcpy(&version[write_loc].value, source, sizeof(ValueType));
    __atomic_store_n(&version[write_loc].version, epoch, __ATOMIC_SEQ_CST);
}

} // namespace epic

#endif // STORAGE_H
