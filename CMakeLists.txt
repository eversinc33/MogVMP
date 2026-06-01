cmake_minimum_required(VERSION 3.21)
project(lifter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Point cmake to remill's pre-built dependencies (XED, gflags, glog, etc.)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/remill/dependencies/install")

# Disable remill's tests and install target when used as subdirectory
set(REMILL_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(REMILL_ENABLE_INSTALL_TARGET OFF CACHE BOOL "" FORCE)

set(ZYDIS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZYAN_ZYCORE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/zydis/dependencies/zycore" CACHE PATH "" FORCE)
add_subdirectory(zydis)

add_subdirectory(remill)

add_executable(lifter
  src/main.cpp
  src/vmp_lifter.cpp
  src/handler_discovery.cpp
  src/instruction_decoder.cpp
  src/peload.cpp
  src/passes/concrete_alloca_prop.cpp
  src/passes/readable_register_names.cpp
  src/passes/readable_block_names.cpp
  src/passes/strip_rdtsc_sideeffects.cpp)

target_link_libraries(lifter PRIVATE remill Zydis::Zydis)

# Suppress GCC 14 false-positive uninitialized warnings from LLVM headers
# (triggered by SmallDenseMap move construction in GVNPass/SROAPass)
target_compile_options(lifter PRIVATE -Wno-maybe-uninitialized)

include(CTest)
if(BUILD_TESTING)
  find_package(GTest REQUIRED)

  add_executable(lifter_tests
    tests/lifter_golden_test.cpp)

  target_compile_definitions(lifter_tests PRIVATE
    LIFTER_BIN="$<TARGET_FILE:lifter>"
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data")

  target_link_libraries(lifter_tests PRIVATE GTest::gtest_main)

  add_test(NAME LifterGoldenTest COMMAND lifter_tests)
endif()
