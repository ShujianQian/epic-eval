//
// Created by Shujian Qian on 2024-04-06.
//


#include <gtest/gtest.h>

#include <benchmarks/tpcc_table.h>

TEST(TpccKeyOrderingTest, CustomerOrderTest)
{
    using key_type = epic::tpcc::ClientOrderKey;
    key_type key1 {1, 2, 3, 4};
    key_type key2 {2, 2, 3, 4};
    ASSERT_GT(key2.base_key, key1.base_key);

    key1 = key_type {1, 3, 3, 4};
    ASSERT_LT(key2.base_key, key1.base_key);
}