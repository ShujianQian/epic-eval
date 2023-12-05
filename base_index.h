//
// Created by Shujian Qian on 2023-11-29.
//

#ifndef EPIC__BASE_INDEX_H
#define EPIC__BASE_INDEX_H

#include <memory>

namespace epic {

template<typename KeyType, typename ValueType, KeyType InvalidKey = std::numeric_limits<KeyType>::max(),
    ValueType InvalidValue = std::numeric_limits<ValueType>::max()>
class BaseIndex
{
public:
    constexpr static KeyType INVALID_KEY = InvalidKey;
    constexpr static ValueType INVALID_VALUE = InvalidValue;
    class Iterator
    {
    public:
        virtual ~Iterator() = default;
        virtual void advance() = 0;
        virtual bool isValid() = 0;
        virtual ValueType getValue() = 0;
        virtual KeyType getKey() = 0;
    };

    virtual ~BaseIndex() = default;

    virtual ValueType searchOrInsert(KeyType key) = 0;
    virtual ValueType searchOrInsert(KeyType key, bool &inserted) = 0;
    virtual ValueType search(KeyType key) = 0;
    virtual std::unique_ptr<Iterator> searchRange(KeyType start, KeyType end) = 0;
    virtual std::unique_ptr<Iterator> searchRangeReverse(KeyType start, KeyType end) = 0;
    virtual std::unique_ptr<Iterator> searchAfter(KeyType start) = 0;
};

} // namespace epic

#endif // EPIC__BASE_INDEX_H
