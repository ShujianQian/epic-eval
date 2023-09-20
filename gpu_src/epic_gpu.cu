//
// Created by Shujian Qian on 2023-07-18.
//

#include "gpu_src/epic_gpu.cuh"

#include <stdio.h>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <x86intrin.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <cub/cub.cuh>

int epic_gpu(int argc, char **argv)
{
    return 0;
}

namespace {

std::atomic<bool> stop_cpu_thread = false;

[[maybe_unused]] int make_cpu_busy()
{

    // The workload size can be adjusted according to the memory capacity.
    // Be aware, a too large value can lead to bad_alloc exception.
    size_t workloadSize = 100'000;

    std::vector<int> workload(workloadSize, 42);
    uint64_t iter_count = 0;
    while (!stop_cpu_thread)
    {
        // Do some memory intensive work, here we are reversing the vector
        std::reverse(workload.begin(), workload.end());
        iter_count++;
    }

    //      std::cout << "CPU thread " << std::this_thread::get_id () << " finished with " << iter_count << " iterations"
    //                << std::endl;
    return 0;
}
} // namespace

int run_thrust_experiment(int argc, char **argv)
{

    if (argc < 5)
    {
        std::cout << "Usage: " << argv[0] << " <num_threads>" << std::endl;
        return 1;
    }

    int num_threads = std::stoi(argv[1]);
    bool use_blocking_schedule = strcmp(argv[2], "true") == 0;
    int start_bit = std::stoi(argv[3]);
    int end_bit = std::stoi(argv[4]);

    uint64_t num_elements = 10'000'000;
    uint64_t *d_data, *d_sorted_data;
    if (use_blocking_schedule)
    {
        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    }

    cudaMalloc((void **)&d_data, num_elements * sizeof(uint64_t));
    cudaMalloc((void **)&d_sorted_data, num_elements * sizeof(uint64_t));

    thrust::device_ptr<uint64_t> dp_data(d_data);

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back(make_cpu_busy);
    }

    thrust::sort(dp_data, dp_data + num_elements);

    auto thrust_start_time = std::chrono::high_resolution_clock::now();

    thrust::sort(dp_data, dp_data + num_elements);
    thrust::sort(dp_data, dp_data + num_elements);
    thrust::sort(dp_data, dp_data + num_elements);
    thrust::sort(dp_data, dp_data + num_elements);
    thrust::sort(dp_data, dp_data + num_elements);
    thrust::sort(dp_data, dp_data + num_elements);
    thrust::sort(dp_data, dp_data + num_elements);
    thrust::sort(dp_data, dp_data + num_elements);
    thrust::sort(dp_data, dp_data + num_elements);
    thrust::sort(dp_data, dp_data + num_elements);

    auto thrust_end_time = std::chrono::high_resolution_clock::now();

    void *temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_elements, start_bit, end_bit);
    std::cout << "temp_storage size: " << temp_storage_bytes / sizeof(uint64_t) << std::endl;
    cudaMalloc((void **)&temp_storage, temp_storage_bytes);

    if (temp_storage == nullptr)
    {
        std::cout << "Failed to allocate temp_storage" << std::endl;
    }

    auto cub_start_time = std::chrono::high_resolution_clock::now();

    cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_elements, start_bit, end_bit);
    cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_elements, start_bit, end_bit);
    cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_elements, start_bit, end_bit);
    cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_elements, start_bit, end_bit);
    cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_elements, start_bit, end_bit);
    cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_elements, start_bit, end_bit);
    cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_elements, start_bit, end_bit);
    cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_elements, start_bit, end_bit);
    cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_elements, start_bit, end_bit);
    cub::DeviceRadixSort::SortKeys(temp_storage, temp_storage_bytes, d_data, d_sorted_data, num_elements, start_bit, end_bit);
    cudaDeviceSynchronize();

    auto cub_end_time = std::chrono::high_resolution_clock::now();

    stop_cpu_thread = true;

    for (auto &thread : threads)
    {
        thread.join();
    }
    auto cpu_end_time = std::chrono::high_resolution_clock::now();

    std::cout << "Thrust Elapsed time: "
              << std::chrono::duration_cast<std::chrono::microseconds>((thrust_end_time - thrust_start_time)).count() << " us" << std::endl;

    std::cout << "CUB Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>((cub_end_time - cub_start_time)).count()
              << " us" << std::endl;

    std::cout << "CPU Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>((cpu_end_time - thrust_start_time)).count()
              << " us" << std::endl;

    return 0;
}