//
// Created by Shujian Qian on 2023-09-12.
//

#ifndef UNORDERED_INDEX_H
#define UNORDERED_INDEX_H

#include <cstdint>

namespace epic {

template<typename KeyType, typename RowIdType = uint32_t>
class UnorderedIndex
{
public:
    virtual ~UnorderedIndex() = default;
    static constexpr RowIdType INVALID_ROW_ID = -1;
    virtual RowIdType findOrInsertRow(KeyType key, uint32_t epoch_id) = 0;
    virtual RowIdType findRow(KeyType key, uint32_t epoch_id) = 0;
    virtual void deleteRow(KeyType key, uint32_t epoch_id) = 0;
};

} // namespace epic

#endif // UNORDERED_INDEX_H
