//
// Created by Shujian Qian on 2023-11-29.
//

#ifndef EPIC__STD_UNORDERED_MAP_INDEX_H
#define EPIC__STD_UNORDERED_MAP_INDEX_H

#include <queue>
#include <limits>
#include <unordered_map>

#include <base_index.h>

namespace epic {

/**
 * This is not a thread-safe implementation.
 *
 * @tparam KeyType
 * @tparam ValueType
 */
template<typename KeyType, typename ValueType, KeyType InvalidKey = std::numeric_limits<KeyType>::max(),
    ValueType InvalidValue = std::numeric_limits<ValueType>::max()>
class StdUnorderedMapIndex : public BaseIndex<KeyType, ValueType, InvalidKey, InvalidValue>
{
    std::queue<ValueType> free_list;
    std::unordered_map<KeyType, ValueType> map;

public:
    using Iterator = typename BaseIndex<KeyType, ValueType, InvalidKey, InvalidValue>::Iterator;
    constexpr static KeyType INVALID_KEY = InvalidKey;
    constexpr static ValueType INVALID_VALUE = InvalidValue;

    explicit StdUnorderedMapIndex(ValueType max_row_id)
        : map(max_row_id)
    {
        for (ValueType i = 0; i < max_row_id; i++)
        {
            free_list.push(i);
        }
    }
    ~StdUnorderedMapIndex() override = default;

    inline ValueType searchOrInsert(KeyType key) override
    {
        ValueType value = free_list.front();
        auto res = map.emplace(std::make_pair(key, value));
        if (res.second)
        {
            free_list.pop();
        }
        return res.first->second;
    }

    inline ValueType searchOrInsert(KeyType key, bool &inserted) override
    {
        ValueType value = free_list.front();
        auto res = map.emplace(std::make_pair(key, value));
        if (res.second)
        {
            free_list.pop();
            inserted = true;
        }
        else
        {
            inserted = false;
        }
        return res.first->second;
    }

    inline ValueType search(KeyType key) override
    {
        auto it = map.find(key);
        if (it == map.end())
        {
            return BaseIndex<KeyType, ValueType>::INVALID_VALUE;
        }
        return it->second;
    }

    inline std::unique_ptr<Iterator> searchRange(KeyType start, KeyType end) override
    {
        throw std::runtime_error("Hashtable index does not support range search.");
        return nullptr;
    }
    inline std::unique_ptr<Iterator> searchRangeReverse(KeyType start, KeyType end) override
    {
        throw std::runtime_error("Hashtable index does not support range search.");
        return nullptr;
    }
    inline std::unique_ptr<Iterator> searchAfter(KeyType start) override
    {
        throw std::runtime_error("Hashtable index does not support range search.");
        return nullptr;
    }
};

} // namespace epic

#endif // EPIC__STD_UNORDERED_MAP_INDEX_H
