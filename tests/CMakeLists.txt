# unit tests
include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)

CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        GIT_TAG release-1.11.0
        VERSION 1.11.0
        OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt"
)

add_executable(test_execution_planner test_execution_planner.cu)
target_link_libraries(test_execution_planner PRIVATE epic gtest_main)

add_executable(test_masstree_index test_masstree_index.cpp)
target_link_libraries(test_masstree_index PRIVATE epic gtest_main)

add_executable(test_tpcc_key_ordering test_tpcc_key_ordering.cpp)
target_link_libraries(test_tpcc_key_ordering PRIVATE epic gtest_main)
