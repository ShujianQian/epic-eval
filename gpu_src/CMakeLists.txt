add_library(epic_gpu SHARED epic_gpu.cu epic_gpu.cuh ../gpu_allocator.cu ../gpu_execution_planner.cu
        ../benchmarks/gpu_ycsb.cu
        ../gpu_txn_bridge.cu
        ../gpu_txn.cu
        ../util_gpu_transfer.cu
        ../benchmarks/tpcc_gpu_submitter.cu
        ../gpu_configuration.cu
        ../benchmarks/tpcc_gpu_executor.cu)
target_include_directories(epic_gpu PUBLIC ${EPIC_INCLUDE_DIR})
set(EPIC_GPU_LINK_LIB ${EPIC_GPU_LINK_LIB} epic_gpu PARENT_SCOPE)
