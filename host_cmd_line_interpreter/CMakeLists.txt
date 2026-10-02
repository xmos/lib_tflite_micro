cmake_minimum_required(VERSION 3.21)

# **********************
# Setup XMOS toolchain
# **********************
project(model_runner VERSION 1.0.0)

enable_language(CXX C)

# **********************
# Disable in-source build.
# **********************
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "In-source build is not allowed! Please specify a build folder.\n\tex:cmake -B build")
endif()

# **********************
# install
# **********************
set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/bin")

# **********************
# Build flags
# **********************

if(MSVC)
  # MSVC Debug builds now omit /RTC1, avoiding its conflict with lib_nn's /O2.
  string(REPLACE "/RTC1" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
  string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  set(BUILD_FLAGS "/O2" "/MP16")
else()
  set(BUILD_FLAGS "-O3")
endif()

list(APPEND BUILD_FLAGS
  "-D__xtflm_conf_h_exists__"
  "-DNN_USE_REF"
)

# **********************
# firmware targets
# **********************
add_executable(xtflm_interpreter_cmdline)
target_compile_features(xtflm_interpreter_cmdline PRIVATE cxx_std_17)
target_compile_options(xtflm_interpreter_cmdline PRIVATE ${BUILD_FLAGS})
find_package(Threads REQUIRED)
target_link_libraries(xtflm_interpreter_cmdline PRIVATE Threads::Threads)

include("${CMAKE_CURRENT_SOURCE_DIR}/../lib_tflite_micro/sources.cmake")

target_link_libraries(xtflm_interpreter_cmdline PRIVATE lib_nn)

target_sources(xtflm_interpreter_cmdline
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc"
    ${ALL_SOURCES}
)

list(APPEND ALL_INCLUDES
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_include_directories(xtflm_interpreter_cmdline
  PRIVATE
    ${ALL_INCLUDES}
)

install(TARGETS xtflm_interpreter_cmdline DESTINATION ${INSTALL_DIR})

# Testing
enable_testing()
set(TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mobnet")
set(TEST_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/test_mobnet_out")
set(MODEL_NAME "${TEST_DIR}/model_mobilenet_v1.tflite")
set(PARAMS_NAME "${TEST_DIR}/model_mobilenet_v1.params")
set(INPUT_NAME "${TEST_DIR}/baboon.raw")
set(CLASSES_NAME "${TEST_DIR}/classes.raw")

add_test(NAME test_smoke COMMAND 
  xtflm_interpreter_cmdline ${MODEL_NAME} ${PARAMS_NAME} ${INPUT_NAME} ${TEST_OUTPUT}
)
add_test(NAME test_smoke_compare COMMAND 
  ${CMAKE_COMMAND} -E compare_files ${TEST_OUTPUT} ${CLASSES_NAME}
)

set_tests_properties(test_smoke PROPERTIES FIXTURES_SETUP smoke_output)
set_tests_properties(test_smoke_compare PROPERTIES FIXTURES_REQUIRED smoke_output)
