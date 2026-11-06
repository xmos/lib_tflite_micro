set(TARGET_NAME hello_world)

# optimize the model with xcore-opt (from the xmos-ai-tools python package)
find_program(XCORE_OPT_EXECUTABLE NAMES xcore-opt REQUIRED)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "XCORE_VX4" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv32")
  set(ARCH_STR "VX4A")
else()
  set(ARCH_STR "XS3A")
endif()
set(MODEL_N_CORES 1)
set(MODEL_IN_PATH "${CMAKE_CURRENT_LIST_DIR}/src/hello_world_int8.tflite")
set(MODEL_OUT_PATH "${CMAKE_CURRENT_LIST_DIR}/src/hello_world_int8_opt.tflite")
execute_process(
  COMMAND ${XCORE_OPT_EXECUTABLE} ${MODEL_IN_PATH} -tc ${MODEL_N_CORES} -o ${MODEL_OUT_PATH} --xcore-target-arch=${ARCH_STR}
  WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
  COMMAND_ERROR_IS_FATAL ANY
)

add_executable(${TARGET_NAME}
  src/main.cpp
  src/hello_world_int8_opt.tflite.cpp
)
target_include_directories(${TARGET_NAME} PRIVATE ${COMMON_INCLUDES})
target_compile_definitions(${TARGET_NAME} PRIVATE ${COMMON_DEFINITIONS})
target_compile_features(${TARGET_NAME} PRIVATE ${COMMON_FEATURES})
target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_OPTIONS} ${XCORE_TARGET})
target_link_options(${TARGET_NAME} PRIVATE ${XCORE_TARGET})
target_link_libraries(${TARGET_NAME} PRIVATE xtflitemicro)

if(NOT NATIVE)
  set_property(TARGET ${TARGET_NAME} PROPERTY CROSSCOMPILING_EMULATOR "${XSIM_EXECUTABLE}")
endif()

add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
set_tests_properties(${TARGET_NAME} PROPERTIES
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
