//
// Created by Shujian Qian on 2024-01-25.
//
#include <benchmarks/tpcc_pver_copyer.h>

#include <chrono>

#include <util_gpu_error_check.cuh>
#include <pver_copyer.cuh>
#include <util_log.h>

#define PRINT_INDIVIDUAL_COPY_TIME 0

namespace epic::tpcc {

void copyTpccPver(TpccRecords records, TpccVersions versions, op_t *d_warehouse_ops_to_copy,
    uint32_t *d_warehouse_ver_to_copy, uint32_t warehouse_num_copy, op_t *d_district_ops_to_copy,
    uint32_t *d_district_ver_to_copy, uint32_t district_num_copy, op_t *d_customer_ops_to_copy,
    uint32_t *d_customer_ver_to_copy, uint32_t customer_num_copy, op_t *d_stock_ops_to_copy,
    uint32_t *d_stock_ver_to_copy, uint32_t stock_num_copy)
{
#ifdef PRINT_INDIVIDUAL_COPY_TIME
    auto &logger = Logger::GetInstance();
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    start_time = std::chrono::high_resolution_clock::now();
#endif
    copyPver(records.warehouse_record, versions.warehouse_version, d_warehouse_ops_to_copy, d_warehouse_ver_to_copy,
        warehouse_num_copy);
#ifdef PRINT_INDIVIDUAL_COPY_TIME
    gpu_err_check(cudaDeviceSynchronize());
    end_time = std::chrono::high_resolution_clock::now();
    logger.Info("Copy warehouse pver time: {} us",
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    start_time = std::chrono::high_resolution_clock::now();
#endif
    copyPver(records.district_record, versions.district_version, d_district_ops_to_copy, d_district_ver_to_copy,
        district_num_copy);
#ifdef PRINT_INDIVIDUAL_COPY_TIME
    gpu_err_check(cudaDeviceSynchronize());
    end_time = std::chrono::high_resolution_clock::now();
    end_time = std::chrono::high_resolution_clock::now();
    logger.Info("Copy district pver time: {} us",
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    start_time = std::chrono::high_resolution_clock::now();
#endif
    copyPver(records.customer_record, versions.customer_version, d_customer_ops_to_copy, d_customer_ver_to_copy,
        customer_num_copy);
#ifdef PRINT_INDIVIDUAL_COPY_TIME
    gpu_err_check(cudaDeviceSynchronize());
    end_time = std::chrono::high_resolution_clock::now();
    end_time = std::chrono::high_resolution_clock::now();
    logger.Info("Copy customer pver time: {} us",
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
    start_time = std::chrono::high_resolution_clock::now();
#endif
    copyPver(records.stock_record, versions.stock_version, d_stock_ops_to_copy, d_stock_ver_to_copy, stock_num_copy);
#ifdef PRINT_INDIVIDUAL_COPY_TIME
    gpu_err_check(cudaDeviceSynchronize());
    end_time = std::chrono::high_resolution_clock::now();
    end_time = std::chrono::high_resolution_clock::now();
    logger.Info("Copy stock pver time: {} us",
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());
#endif

    gpu_err_check(cudaDeviceSynchronize());
}

} // namespace epic::tpcc