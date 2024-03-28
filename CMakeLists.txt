cmake_minimum_required(VERSION 3.18)

project(epic CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA QUIET)

if (CUDA_FOUND)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    enable_language(CUDA)
    add_definitions(-DEPIC_CUDA_AVAILABLE)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
else ()
    message(WARNING "CUDA not found, using CPU only. CUDA Related modules will not be available.")
endif ()

set(EPIC_CPU_SOURCE
        util_log.cpp
        util_math.cpp
        benchmarks/ycsb.cpp
        benchmarks/tpcc_txn.cpp
        benchmarks/tpcc.cpp
        benchmarks/tpcc_index.cpp
        benchmarks/tpcc_txn_gen.cpp
        txn_bridge.cpp
        benchmarks/tpcc_cpu_executor.cpp
        benchmarks/ycsb_cpu_executor.cpp)

set(GACCO_CPU_SOURCE
        gacco/benchmarks/tpcc.cpp
        gacco/benchmarks/tpcc_txn_gen.cpp
        gacco/benchmarks/ycsb.cpp)

set(EPIC_GPU_SOURCE
        gpu_src/epic_gpu.cu gpu_allocator.cu
        gpu_execution_planner.cu
        benchmarks/gpu_ycsb.cu
        gpu_txn_bridge.cu
        gpu_txn.cu
        util_gpu_transfer.cu
        benchmarks/tpcc_gpu_submitter.cu
        gpu_configuration.cu
        benchmarks/tpcc_gpu_executor.cu
        gpu_hashtable_index.cu
        util_memory.cu
        benchmarks/tpcc_gpu_index.cu
        benchmarks/ycsb_gpu_index.cu
        benchmarks/ycsb_gpu_submitter.cu
        benchmarks/ycsb_gpu_executor.cu
        pver_copyer.cu
        benchmarks/ycsb_pver_copyer.cu
        benchmarks/tpcc_pver_copyer.cu)

set(GACCO_GPU_SOURCE
    gacco/gpu_execution_planner.cu
        gacco/benchmarks/tpcc_gpu_submitter.cu
        gacco/benchmarks/tpcc_gpu_executor.cu
        gacco/benchmarks/ycsb_gpu_submitter.cu
        gacco/benchmarks/ycsb_gpu_executor.cu)

if (CUDA_FOUND)
    set(EPIC_SOURCE ${EPIC_CPU_SOURCE} ${EPIC_GPU_SOURCE} ${GACCO_CPU_SOURCE} ${GACCO_GPU_SOURCE})
else ()
    set(EPIC_SOURCE ${EPIC_CPU_SOURCE} ${GACCO_CPU_SOURCE})
endif ()

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/gpu_src ${CMAKE_SOURCE_DIR}/third_party/spdlog/include)
if (CUDA_FOUND)
    include_directories(${CMAKE_SOURCE_DIR}/third_party/cuco/include)
endif()
add_library(epic STATIC ${EPIC_SOURCE})
if (CUDA_FOUND)
    set_property(TARGET epic PROPERTY CUDA_ARCHITECTURES 86)
endif ()

add_executable(epic_driver main.cpp)
target_link_libraries(epic_driver epic)

if (CUDA_FOUND)
    add_executable(test_epic ${CMAKE_SOURCE_DIR}/tests/unit_tests/test_gpu_table.cu)
    target_link_libraries(test_epic epic)
endif ()

#set(EPIC_INCLUDE_DIR ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/gpu_src)
#set(LIBEPIC_SOURCE)
#
#set(EPIC_LINK_LIB epic_main)
#set(EPIC_GPU_LINK_LIB)
#
#add_subdirectory(third_party)
#
#add_library(epic_main SHARED util_log.cpp util_math.cpp
#        benchmarks/ycsb.cpp
#        benchmarks/tpcc_txn.cpp
#        benchmarks/tpcc.cpp
#        benchmarks/tpcc_index.cpp
#        txn_bridge.cpp)
#target_include_directories(epic_main PUBLIC ${EPIC_INCLUDE_DIR})
#
#add_executable(epic main.cpp)
#target_include_directories(epic PUBLIC ${EPIC_INCLUDE_DIR})
#
#if (CUDA_FOUND)
#    add_subdirectory(gpu_src)
#    target_link_libraries(epic_gpu PRIVATE ${EPIC_LINK_LIB})
#    target_link_libraries(epic PRIVATE ${EPIC_GPU_LINK_LIB})
#endif ()
#
#target_link_libraries(epic PRIVATE ${EPIC_LINK_LIB})
#
#add_subdirectory(tests)
