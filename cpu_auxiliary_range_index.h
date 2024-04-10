//
// Created by Shujian Qian on 2024-03-31.
//

#ifndef CPU_AUXILIARY_RANGE_INDEX_H
#define CPU_AUXILIARY_RANGE_INDEX_H

#include <limits>
#include <thread>
#include <unistd.h>
#include <type_traits>

#include <build/config.h>
#include <kvthread.hh>
#include <masstree_insert.hh>
#include <masstree_remove.hh>
#include <masstree_tcursor.hh>
#include <masstree_print.hh>
#include <masstree_scan.hh>
#include <kvthread.hh>
#include <timestamp.hh>
#include <masstree.hh>

#include <util_endianness.h>
#include <util_log.h>

namespace epic {

template<typename ValueType>
struct CpuAuxRangeIndexParam : public Masstree::nodeparams<>
{

    typedef ValueType value_type;
    typedef threadinfo threadinfo_type;
};

extern thread_local threadinfo *ti;

template<typename KeyType, typename ValueType>
class CpuAuxRangeIndex : protected Masstree::basic_table<CpuAuxRangeIndexParam<ValueType>>
{
    using base_table_type = Masstree::basic_table<CpuAuxRangeIndexParam<ValueType>>;

public:
    using key_type = KeyType;
    using value_type = ValueType;

    static_assert(std::is_integral_v<KeyType>, "KeyType must be an integral type");
    static_assert(std::is_integral_v<ValueType>, "ValueType must be an integral type");
    static_assert(std::is_unsigned_v<KeyType>, "KeyType must be an unsigned type");

    static constexpr ValueType invalid_value = std::numeric_limits<ValueType>::max();

    CpuAuxRangeIndex()
        : base_table_type()
    {
        auto &logger = Logger::GetInstance();
        logger.Info("initializing CpuAuxRangeIndex");
        this->initialize(*GetThreadInfo());
    }

    static threadinfo *GetThreadInfo()
    {
        if (ti == nullptr)
        {
            auto &logger = Logger::GetInstance();
            logger.Info("getting thread id {}", gettid());
            ti = threadinfo::make(threadinfo::TI_MAIN, gettid());
        }
        return ti;
    }
    template<typename BaseIteratorType>
    class iterator : public BaseIteratorType
    {
        KeyType curr_key;
        ValueType curr_value;
        KeyType end_key;

    public:
        iterator(typename BaseIteratorType::node_type *root, lcdf::Str firstkey, threadinfo &ti)
            : BaseIteratorType(root, firstkey, ti)
            , end_key(std::numeric_limits<KeyType>::max())
        {}

        void adapt()
        {
            if (!this->terminated)
            {
                KeyType bs_curr_key = *reinterpret_cast<const KeyType *>(this->key.full_string().data());
                curr_key = endian_swap(bs_curr_key);
                curr_value = this->entry.value();
            }
        }

        void get_next()
        {
            this->next(*GetThreadInfo());
            adapt();
        }

        bool is_valid() const
        {
            static_assert(std::is_same_v<BaseIteratorType, typename base_table_type::forward_scan_iterator_impl> ||
                              std::is_same_v<BaseIteratorType, typename base_table_type::reverse_scan_iterator_impl>,
                "iterator type not supported");
            if constexpr (std::is_same_v<BaseIteratorType, typename base_table_type::forward_scan_iterator_impl>)
            {
                return !this->terminated && !(end_key < curr_key);
            }
            else
            {
                return !this->terminated && !(curr_key < end_key);
            }
        }

        void set_iterator_end_key(KeyType end)
        {
            end_key = end;
        }

        key_type getKey() const
        {
            return curr_key;
        }

        value_type getValue() const
        {
            return curr_value;
        }
    };

    using forward_iterator = iterator<typename base_table_type::forward_scan_iterator_impl>;
    using reverse_iterator = iterator<typename base_table_type::reverse_scan_iterator_impl>;

    ValueType searchOrInsert(KeyType key, ValueType value)
    {
        KeyType bs_key = endian_swap(key);
        typename base_table_type::cursor_type cursor(*this, reinterpret_cast<char *>(&bs_key), sizeof(KeyType));
        auto &ti = *GetThreadInfo();
        bool found = cursor.find_insert(ti);
        if (!found)
        {
            cursor.value() = value;
        }
        ValueType retval = cursor.value();
        cursor.finish(1, ti);
        return retval;
    }

    ValueType search(KeyType key)
    {
        auto &ti = *GetThreadInfo();
        KeyType bs_key = endian_swap(key);
        ValueType result = invalid_value;
        this->get(lcdf::Str(reinterpret_cast<char *>(&bs_key), sizeof(KeyType)), result, ti);
        return result;
    }

    forward_iterator *searchForwardIterator(KeyType start, KeyType end)
    {
        KeyType bs_start = endian_swap(start);
        auto &ti = *GetThreadInfo();
        auto it = this->template find_iterator<forward_iterator>(
            lcdf::Str(reinterpret_cast<char *>(&bs_start), sizeof(KeyType)), ti);
        it->set_iterator_end_key(end);
        it->adapt();
        return it;
    }

    reverse_iterator *searchReverseIterator(KeyType start, KeyType end)
    {
        KeyType bs_start = endian_swap(start);
        auto &ti = *GetThreadInfo();
        auto it = this->template find_iterator<reverse_iterator>(
            lcdf::Str(reinterpret_cast<char *>(&bs_start), sizeof(KeyType)), ti);
        it->set_iterator_end_key(end);
        it->adapt();
        return it;
    }
};

} // namespace epic

#endif // CPU_AUXILIARY_RANGE_INDEX_H
