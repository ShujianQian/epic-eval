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
    set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
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
        txn_bridge.cpp)

set(EPIC_GPU_SOURCE
        gpu_src/epic_gpu.cu gpu_src/gpu_allocator.cu
        gpu_execution_planner.cu
        benchmarks/gpu_ycsb.cu
        gpu_txn_bridge.cu
        gpu_txn.cu
        util_gpu_transfer.cu
        benchmarks/tpcc_gpu_submitter.cu
        gpu_configuration.cu
        benchmarks/tpcc_gpu_executor.cu)

if (CUDA_FOUND)
    set(EPIC_SOURCE ${EPIC_CPU_SOURCE} ${EPIC_GPU_SOURCE})
else ()
    set(EPIC_SOURCE ${EPIC_CPU_SOURCE})
endif ()

include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/gpu_src ${CMAKE_SOURCE_DIR}/third_party/spdlog/include)
add_library(epic STATIC ${EPIC_SOURCE})

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
