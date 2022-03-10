cmake_minimum_required(VERSION 3.14)

#**********************
# Setup XMOS toolchain
#**********************
#include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/xmos_toolchain.cmake")

project(model_runner VERSION 1.0.0)

enable_language(CXX C)

#**********************
# Disable in-source build.
#**********************
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source build is not allowed! Please specify a build folder.\n\tex:cmake -B build")
endif()

#**********************
# install
#**********************
set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/bin")

#**********************
# Build flags
#**********************

set(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "C++ Compiler Base Flags" FORCE)

set(BUILD_FLAGS
  "-O3"
  "-DNN_USE_REF"
  "-D__xtflm_conf_h_exists__"
  "-fsanitize=undefined" 
  "-fsanitize=integer"
  "-fsanitize=implicit-conversion"
  "-fsanitize=address"
  "-fsanitize-recover=address" 

)

#**********************
# firmware targets
#**********************
add_executable(tflm_interpreter_cmdline)
target_compile_options(tflm_interpreter_cmdline PRIVATE ${BUILD_FLAGS})
target_link_options(tflm_interpreter_cmdline PRIVATE ${BUILD_FLAGS})
target_link_libraries(tflm_interpreter_cmdline PRIVATE "-lpthread")

include(../cmakefiles/xtflm.cmake)

target_sources(tflm_interpreter_cmdline
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/xcore_strided_slice_test.cc"
  PRIVATE ${ALL_SOURCES}
)

target_include_directories(tflm_interpreter_cmdline
  PRIVATE ${ALL_INCLUDES}
)

install(TARGETS tflm_interpreter_cmdline DESTINATION ${INSTALL_DIR})
