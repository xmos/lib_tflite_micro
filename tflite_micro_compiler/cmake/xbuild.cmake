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

set(BUILD_FLAGS
  "-g"
  "-O0"
)

#**********************
# Targets
#**********************
set(TOP_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/..")
include(${TOP_DIR}/cmakefiles/xtflm.cmake)

add_library(xtflitemicro SHARED)
set(DEFINTIONS
  "__xtflm_conf_h_exists__"
  "NO_INTERPRETER"
  "NN_USE_REF"
  "TF_LITE_STATIC_MEMORY"
  "TF_LITE_DISABLE_X86_NEON"
)
target_compile_options(xtflitemicro PRIVATE ${BUILD_FLAGS})
target_link_options(xtflitemicro PRIVATE ${BUILD_FLAGS})
target_compile_definitions(xtflitemicro PUBLIC
    ${DEFINTIONS}
)
target_compile_features(xtflitemicro PUBLIC cxx_std_11)
target_sources(xtflitemicro
  PRIVATE ${TFLM_KERNEL_SOURCES}
  PRIVATE ${TFLITE_SOURCES}
  PRIVATE ${NN_SOURCES}
  PRIVATE ${XTFLIB_KERNEL_SOURCES}
)
target_include_directories(xtflitemicro
  PRIVATE ${ALL_INCLUDES}
)
install(TARGETS xtflitemicro DESTINATION ${INSTALL_DIR})


add_executable(tflite_micro_compiler)
unset(DEFINTIONS)
set(DEFINTIONS
  "__xtflm_conf_h_exists__"
  "NN_USE_REF"
  "TF_LITE_STATIC_MEMORY"
  "TF_LITE_DISABLE_X86_NEON"
  "SUFFICIENT_ARENA_SIZE=128*1024*1024"
)
target_compile_options(tflite_micro_compiler PRIVATE ${BUILD_FLAGS})
target_link_options(tflite_micro_compiler PRIVATE ${BUILD_FLAGS})
file(GLOB_RECURSE COMPILER_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h")
file(GLOB_RECURSE COMPILER_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
target_compile_definitions(tflite_micro_compiler PUBLIC
    ${DEFINTIONS}
)
target_compile_features(tflite_micro_compiler PUBLIC cxx_std_11)
target_sources(tflite_micro_compiler
  PRIVATE ${ALL_SOURCES}
  PRIVATE ${COMPILER_SRCS}
)
target_include_directories(tflite_micro_compiler
  PRIVATE ${COMPILER_HEADERS}
  PRIVATE ${ALL_INCLUDES}
)
install(TARGETS tflite_micro_compiler DESTINATION ${INSTALL_DIR})
