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
  "-D__xtflm_conf_h_exists__"
  "-DNN_USE_REF"
)

#**********************
# firmware targets
#**********************
add_executable(xtflm_interpreter_cmdline)
target_compile_options(xtflm_interpreter_cmdline PRIVATE ${BUILD_FLAGS})
target_link_options(xtflm_interpreter_cmdline PRIVATE ${BUILD_FLAGS})
target_link_libraries(xtflm_interpreter_cmdline PRIVATE "-lpthread")

set(TOP_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/..")

include(../cmakefiles/xtflm.cmake)

target_sources(xtflm_interpreter_cmdline
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc"
  PRIVATE ${ALL_SOURCES}
)

target_include_directories(xtflm_interpreter_cmdline
  PRIVATE ${ALL_INCLUDES}
)

install(TARGETS xtflm_interpreter_cmdline DESTINATION ${INSTALL_DIR})
