set(TARGET_NAME hello_world)

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
