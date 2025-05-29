# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(execute_sinh_op)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../run/out")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../outputs")

set(INC_PATH $ENV{DDK_PATH})
set(AUTO_GEN_PATH "../SinhCustom/build_out/autogen")

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(LIB_PATH $ENV{NPU_HOST_LIB})

# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub/")
    set(LIB_PATH1 "/usr/local/Ascend/ascend-toolkit/latest/atc/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

# Header path
include_directories(
    ${INC_PATH}/runtime/include
    ${INC_PATH}/atc/include
    ../inc
    ${AUTO_GEN_PATH}
)

# add host lib path
link_directories(
    ${LIB_PATH}
    ${LIB_PATH1}
)

add_executable(execute_sinh_op
    ${AUTO_GEN_PATH}/aclnn_sinh_custom.cpp
    operator_desc.cpp
    op_runner.cpp
    main.cpp
    op_runner.cpp
    common.cpp
)

target_link_libraries(execute_sinh_op
    ascendcl
    acl_op_compiler
    nnopbase
    stdc++
)

install(TARGETS execute_sinh_op DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
