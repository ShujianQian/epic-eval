//
// Created by Shujian Qian on 2023-09-12.
//

#ifndef HASHTABLE_INDEX_H
#define HASHTABLE_INDEX_H

#include <unordered_map>
#include <list>
#include <stack>

#include "unordered_index.h"

namespace epic {

/**
 * This is not a thread-safe implementation.
 *
 * @tparam KeyType
 * @tparam RowIdType
 */
template<typename KeyType, typename RowIdType = uint32_t>
class StdHashtableIndex : public UnorderedIndex<KeyType, RowIdType>
{
private:
    std::stack<RowIdType> free_list;
    struct Element
    {
        RowIdType row_id;
        uint32_t epoch_id;
    };
    std::unordered_map<typename KeyType::baseType, Element> map;

public:
    explicit StdHashtableIndex(RowIdType max_row_id)
    : map(max_row_id)
    {
        for (RowIdType i = 0; i < max_row_id; i++)
        {
            free_list.push(i);
        }
    }

    RowIdType findOrInsertRow(KeyType key, uint32_t epoch_id) override
    {
        Element element {free_list.top(), epoch_id};

        auto ret = map.emplace(std::make_pair(key.base_key, element));
        if (ret.second)
        {
            free_list.pop();
        }
        if (ret.first->second.epoch_id != epoch_id)
        {
            ret.first->second.epoch_id = epoch_id;
        }
        return ret.first->second.row_id;
    }

    RowIdType findRow(KeyType key, uint32_t epoch_id) override
    {
        auto it = map.find(key.base_key);
        if (it == map.end())
        {
            return UnorderedIndex<KeyType, RowIdType>::INVALID_ROW_ID;
        }
        if (it->second.epoch_id != epoch_id)
        {
            it->second.epoch_id = epoch_id;
        }
        return it->second.row_id;
    }

    void deleteRow(KeyType key, uint32_t epoch_id) override
    {
        auto it = map.find(key.base_key);
        if (it == map.end())
        {
            return;
        }
        if (it->second.epoch_id > epoch_id)
        {
            return;
        }
        free_list.push(it->second.row_id);
        map.erase(it);
    }
};

} // namespace epic

#endif // HASHTABLE_INDEX_H
