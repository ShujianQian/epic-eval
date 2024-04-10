//
// Created by Shujian Qian on 2024-04-01.
//

#include <gtest/gtest.h>

#include <vector>
#include <thread>
#include <chrono>
#include <cstdint>

#include <cpu_auxiliary_range_index.h>
#include <util_endianness.h>
#include <util_log.h>

TEST(MasstreeIndex, IntKeyCompare)
{
    uint32_t x = 0x11223344u;
    uint32_t y = 0x11332211u;
    uint32_t x_swapped = epic::endian_swap(x);
    uint32_t y_swapped = epic::endian_swap(y);
    int retval = strcmp(reinterpret_cast<char *>(&x_swapped), reinterpret_cast<char *>(&y_swapped));
    ASSERT_LT(retval, 0);

    x = 0xffffffffu;
    y = 0x00000000u;
    x_swapped = epic::endian_swap(x);
    y_swapped = epic::endian_swap(y);
    retval = strcmp(reinterpret_cast<char *>(&x_swapped), reinterpret_cast<char *>(&y_swapped));
    ASSERT_GT(retval, 0);
}

using implementations_to_test =
    testing::Types<epic::CpuAuxRangeIndex<uint32_t, uint32_t>, epic::CpuAuxRangeIndex<uint64_t, uint64_t>,
        epic::CpuAuxRangeIndex<uint32_t, uint64_t>, epic::CpuAuxRangeIndex<uint64_t, uint32_t>>;

template<typename MasstreeIndexType>
class MasstreeIndexTest : public testing::Test
{
public:
    static constexpr size_t num_input = 100000;
    static constexpr size_t num_threads = 16;
    using key_type = typename MasstreeIndexType::key_type;
    using value_type = typename MasstreeIndexType::value_type;
    MasstreeIndexTest()
        : existing_keys(num_input)
        , existing_values(num_input)
        , non_existing_keys(num_input)
    {
        for (size_t i = 0; i < num_input; i++)
        {
            existing_keys[i] = i;
            existing_values[i] = i * 2;
            non_existing_keys[i] = i + num_input;
        }
    }

    std::vector<key_type> existing_keys;
    std::vector<value_type> existing_values;
    std::vector<key_type> non_existing_keys;
};

TYPED_TEST_SUITE(MasstreeIndexTest, implementations_to_test);

TYPED_TEST(MasstreeIndexTest, InsertSearchTest)
{
    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    epic::CpuAuxRangeIndex<key_type, value_type> idx;
    for (int i = 0; i < this->num_input; i++)
    {
        idx.searchOrInsert(this->existing_keys[i], this->existing_values[i]);
    }
    for (int i = 0; i < this->num_input; i++)
    {
        value_type value = idx.search(this->existing_keys[i]);
        ASSERT_EQ(value, this->existing_values[i]);
        value = idx.search(this->non_existing_keys[i]);
        ASSERT_EQ(value, idx.invalid_value);
    }
}

TYPED_TEST(MasstreeIndexTest, ParallelInsertSearchTest)
{
    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    epic::CpuAuxRangeIndex<key_type, value_type> idx;

    std::vector<std::thread> insert_threads;
    insert_threads.reserve(this->num_threads);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < this->num_threads; i++)
    {
        insert_threads.emplace_back([this, &idx, i]() {
            const size_t start = i * this->num_input / this->num_threads;
            const size_t end =
                i == this->num_threads - 1 ? this->num_input : (i + 1) * this->num_input / this->num_threads;
            for (size_t j = start; j < end; ++j)
            {
                idx.searchOrInsert(this->existing_keys[j], this->existing_values[j]);
            }
            /* read my own inserts */
            for (size_t j = start; j < end; ++j)
            {
                value_type value = idx.search(this->existing_keys[j]);
                ASSERT_EQ(value, this->existing_values[j]);
                value = idx.search(this->non_existing_keys[j]);
                ASSERT_EQ(value, idx.invalid_value);
            }
        });
    }
    for (auto &t : insert_threads)
    {
        t.join();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto &logger = epic::Logger::GetInstance();
    logger.Info("Time taken for parallel insert: {} us",
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    for (int i = 0; i < this->num_input; i++)
    {
        value_type value = idx.search(this->existing_keys[i]);
        ASSERT_EQ(value, this->existing_values[i]);
        value = idx.search(this->non_existing_keys[i]);
        ASSERT_EQ(value, idx.invalid_value);
    }
}

TYPED_TEST(MasstreeIndexTest, SearchIteratorTest)
{
    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    epic::CpuAuxRangeIndex<key_type, value_type> idx;
    for (int i = 0; i < this->num_input; i++)
    {
        idx.searchOrInsert(this->existing_keys[i], this->existing_values[i]);
    }

    for (int i = 0; i < this->num_input / 10; ++i)
    {
        int j = i * 10;
        const key_type max_key = std::numeric_limits<key_type>::max();
        auto it = idx.searchForwardIterator(this->existing_keys[j], max_key);

        for (int k = 0; k < 10; ++k)
        {
            ASSERT_TRUE(it->is_valid());
            key_type key = it->getKey();
            ASSERT_EQ(key, this->existing_keys[j + k]);
            value_type value = it->getValue();
            ASSERT_EQ(value, this->existing_values[j + k]);
            it->get_next();
        }
        delete it;

        it = idx.searchForwardIterator(this->non_existing_keys[j], max_key);
        ASSERT_FALSE(it->is_valid());
        delete it;
    }
}

TYPED_TEST(MasstreeIndexTest, SearchReverseIteratorTest)
{
    using key_type = typename TestFixture::key_type;
    using value_type = typename TestFixture::value_type;
    epic::CpuAuxRangeIndex<key_type, value_type> idx;
    for (int i = 0; i < this->num_input; i++)
    {
        idx.searchOrInsert(this->existing_keys[i], this->existing_values[i]);
    }

    for (int i = 0; i < this->num_input / 10; ++i)
    {
        int j = i * 10 + 9;
        const key_type min_key = 0;
        auto it = idx.searchReverseIterator(this->existing_keys[j], min_key);

        for (int k = 0; k < 10; ++k)
        {
            ASSERT_TRUE(it->is_valid());
            key_type key = it->getKey();
            ASSERT_EQ(key, this->existing_keys[j - k]);
            value_type value = it->getValue();
            ASSERT_EQ(value, this->existing_values[j - k]);
            it->get_next();
        }
        delete it;

        it = idx.searchReverseIterator(this->non_existing_keys[j], min_key);
        key_type key = it->getKey();
        ASSERT_EQ(key, this->existing_keys.back());
        value_type value = it->getValue();
        ASSERT_EQ(value, this->existing_values.back());
        delete it;
    }
}