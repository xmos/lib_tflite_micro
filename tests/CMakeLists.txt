cmake_minimum_required(VERSION 3.21)

project(tflite_micro_tests C CXX ASM)

set(SANBOX_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
set(TFLITE_MICRO_DIR "${SANBOX_DIR}/lib_tflite_micro")

# define tflite micro archive
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "XCORE_XS3A")
  set(TFLITE_MICRO_ARCHIVE "libxtflitemicro_xs3a.a")
  set(XCORE_TARGET "-target=XK-EVK-XU316")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "XCORE_VX4")
  set(TFLITE_MICRO_ARCHIVE "libxtflitemicro_vx4b.a")
  set(XCORE_TARGET "-target=XK-EVK-XU416")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv32")
  set(TFLITE_MICRO_ARCHIVE "libxtflitemicro_rvx4b.a")
  set(XCORE_TARGET "")
else()
  set(TFLITE_MICRO_ARCHIVE "libxtflitemicro.a")
  set(XCORE_TARGET "")
  set(NATIVE TRUE)
endif()

# check binary exists
if(NOT EXISTS "${TFLITE_MICRO_DIR}/lib/${TFLITE_MICRO_ARCHIVE}")
  message(FATAL_ERROR "TFLite Micro archive not found: ${TFLITE_MICRO_DIR}/lib/${TFLITE_MICRO_ARCHIVE}")
else()
  message(STATUS "TFLite Micro archive found: ${TFLITE_MICRO_ARCHIVE}")
endif()

# add the library
add_library(xtflitemicro STATIC IMPORTED)
set_target_properties(xtflitemicro PROPERTIES
  IMPORTED_LOCATION "${TFLITE_MICRO_DIR}/lib/${TFLITE_MICRO_ARCHIVE}"
)
set(COMMON_DEFINITIONS TF_LITE_STATIC_MEMORY NO_INTERPRETER)
if(NATIVE)
  set(COMMON_FEATURES cxx_std_17)
else()
  set(COMMON_FEATURES "")
endif()
set(COMMON_INCLUDES
  "${TFLITE_MICRO_DIR}"
  "${SANBOX_DIR}/lib_nn"
  "${TFLITE_MICRO_DIR}/lib_tflite_micro/submodules/tflite-micro"
  "${TFLITE_MICRO_DIR}/lib_tflite_micro/submodules/gemmlowp"
  "${TFLITE_MICRO_DIR}/lib_tflite_micro/submodules/flatbuffers/include"
)
if(MSVC)
  set(COMMON_OPTIONS /O2 /Zi /W3 /UNDEBUG)
else()
  set(COMMON_OPTIONS -Os -g -Wall -UNDEBUG)
endif()

if(NOT NATIVE)
  find_program(XSIM_EXECUTABLE NAMES xsim REQUIRED)
endif()

enable_testing()
add_subdirectory(avg_pool avg_pool_test)
add_subdirectory(hello_world)
