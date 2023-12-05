//
// Created by Shujian Qian on 2023-11-29.
//

#ifndef EPIC__STD_MAP_INDEX_H
#define EPIC__STD_MAP_INDEX_H

#include <map>
#include <queue>
#include <limits>

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
class StdMapIndex : public BaseIndex<KeyType, ValueType, InvalidKey, InvalidValue>
{
    std::queue<ValueType> free_list;
    std::map<KeyType, ValueType> map;

    using StdMapIter = typename std::map<KeyType, ValueType>::iterator;
    using StdMapRIter = typename std::map<KeyType, ValueType>::reverse_iterator;

public:
    template<typename BaseIterType>
    class Iterator : public BaseIndex<KeyType, ValueType, InvalidKey, InvalidValue>::Iterator
    {
        BaseIterType current, end;

    public:
        Iterator(BaseIterType current, BaseIterType end)
            : current(current)
            , end(end)
        {}
        ~Iterator() override = default;
        bool isValid() override
        {
            return current != end;
        }
        void advance() override
        {
            if (isValid())
            {
                ++current;
            }
        }
        KeyType getKey() override
        {
            return current->first;
        }
        ValueType getValue() override
        {
            return current->second;
        }
    };

    constexpr static KeyType INVALID_KEY = InvalidKey;
    constexpr static ValueType INVALID_VALUE = InvalidValue;

    explicit StdMapIndex(KeyType max_key)
    {
        for (ValueType i = 0; i < max_key; ++i)
        {
            free_list.push(i);
        }
    }
    ~StdMapIndex() override = default;

    ValueType searchOrInsert(KeyType key) override
    {
        if (free_list.empty())
        {
            throw std::runtime_error("No more free row id");
        }
        ValueType value = free_list.front();
        auto res = map.emplace(std::make_pair(key, value));
        if (res.second)
        {
            free_list.pop();
        }
        return res.first->second;
    }
    ValueType searchOrInsert(KeyType key, bool &inserted) override
    {
        if (free_list.empty())
        {
            throw std::runtime_error("No more free row id");
        }
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
    ValueType search(KeyType key) override
    {
        auto it = map.find(key);
        if (it == map.end())
        {
            return BaseIndex<KeyType, ValueType>::INVALID_VALUE;
        }
        return it->second;
    }
    std::unique_ptr<typename BaseIndex<KeyType, ValueType, InvalidKey, InvalidValue>::Iterator> searchRange(
        KeyType start, KeyType end) override
    {
        StdMapIter it = map.lower_bound(start);
        StdMapIter end_it = map.upper_bound(end);
        return std::unique_ptr<typename BaseIndex<KeyType, ValueType, InvalidKey, InvalidValue>::Iterator>(
            new Iterator<StdMapIter>(it, end_it));
//        return std::make_unique<Iterator>(it, end_it);
    }
    std::unique_ptr<typename BaseIndex<KeyType, ValueType, InvalidKey, InvalidValue>::Iterator> searchRangeReverse(
        KeyType start, KeyType end) override
    {
        StdMapRIter it = StdMapRIter(map.upper_bound(start));
        StdMapRIter end_it = StdMapRIter(map.lower_bound(end));
        return std::unique_ptr<typename BaseIndex<KeyType, ValueType, InvalidKey, InvalidValue>::Iterator>(
            new Iterator<StdMapRIter>(it, end_it));
//        return std::make_unique<Iterator<StdMapRIter>>(it, end_it);
    }
    std::unique_ptr<typename BaseIndex<KeyType, ValueType, InvalidKey, InvalidValue>::Iterator> searchAfter(
        KeyType start) override
    {
        StdMapIter it = map.upper_bound(start);
        StdMapIter end_it = map.end();
        return std::unique_ptr<typename BaseIndex<KeyType, ValueType, InvalidKey, InvalidValue>::Iterator>(
            new Iterator<StdMapIter>(it, end_it));
//        return std::make_unique<Iterator<StdMapRIter>>(it, end_it);
    }
};

} // namespace epic

#endif // EPIC__STD_MAP_INDEX_H
