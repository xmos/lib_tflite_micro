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
  INTERFACE_COMPILE_DEFINITIONS TF_LITE_STATIC_MEMORY
)

# Add executable
set(TARGET_NAME avg_pool)
set(COMMON_OPTIONS -Os -g -Wall)

add_executable(${TARGET_NAME} avg_pool/main.cpp)
target_include_directories(${TARGET_NAME} PRIVATE
  "${TFLITE_MICRO_DIR}/lib_tflite_micro/submodules/tflite-micro"
  "${TFLITE_MICRO_DIR}/lib_tflite_micro/submodules/gemmlowp"
)
target_link_libraries(${TARGET_NAME} PRIVATE xtflitemicro)

target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_OPTIONS} ${XCORE_TARGET})
target_link_options(${TARGET_NAME} PRIVATE ${XCORE_TARGET})

if(NOT NATIVE)
  find_program(XSIM_EXECUTABLE NAMES xsim REQUIRED)
  set_property(TARGET ${TARGET_NAME} PROPERTY CROSSCOMPILING_EMULATOR "${XSIM_EXECUTABLE}")
endif()

# Add tests
enable_testing()
add_test(NAME avg_pool COMMAND ${TARGET_NAME})
