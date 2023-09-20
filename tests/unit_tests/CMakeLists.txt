
include_directories(${EPIC_INCLUDE_DIR})

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

message(STATUS "EPIC_LINK_LIB: ${EPIC_LINK_LIB}")

#add_library(test_library SHARED ${CMAKE_SOURCE_DIR}/gpu_table.cu ${CMAKE_SOURCE_DIR}/util_log.cpp ${CMAKE_SOURCE_DIR}/util_math.cpp ${CMAKE_SOURCE_DIR}/gpu_src/epic_gpu.cu ${CMAKE_SOURCE_DIR}/gpu_src/gpu_allocator.cu)

if (CUDA_FOUND)
    add_executable(test_gpu_table test_gpu_table.cu)
    #    target_link_libraries(test_gpu_table PRIVATE test_library)
    target_link_libraries(test_gpu_table PRIVATE ${EPIC_LINK_LIB})
    target_link_libraries(test_gpu_table PRIVATE ${EPIC_GPU_LINK_LIB})
endif ()

add_executable(test_util_bitfield test_util_bitfield.cpp)
target_link_libraries(test_util_bitfield PRIVATE ${EPIC_LINK_LIB})
