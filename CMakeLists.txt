cmake_minimum_required(VERSION 3.23)

# **********************
# Disable in-source build.
# **********************
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source build is not allowed! Please specify a build folder.\n\tex:cmake -B build")
endif()

# **********************
# Targets
# **********************
set(TOP_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}")
include(${TOP_DIR}/cmakefiles/xtflm.cmake)

add_library(xtflitemicro)

set(DEFINTIONS
    "__xtflm_conf_h_exists__"
    "NO_INTERPRETER"
    "TF_LITE_STATIC_MEMORY"
    "TF_LITE_DISABLE_X86_NEON"
    "TF_LITE_STRIP_ERROR_STRINGS"
    "PRINT_PROFILER_SUMMARY"
)

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL XCORE_XS3A)
    # # Set the target
    set(XCORE_TARGET "XCORE-AI-EXPLORER")

    enable_language(CXX C ASM)

    set(BUILD_FLAGS
        "-g"
        "-O3"
        "-fcomment-asm"
        "-mcmodel=large"
        "-fxscope"
        "-Wno-xcore-fptrgroup"
        "-Wno-unknown-pragmas"
        "-Wno-bidirectional-buffered-port"
    )
else()
    set(BUILD_FLAGS
        "-g"
        "-O0"
    )

    list(APPEND DEFINTIONS "NN_USE_REF")

    target_compile_features(xtflitemicro PUBLIC cxx_std_11)
endif()

target_compile_options(xtflitemicro PRIVATE ${BUILD_FLAGS})
target_link_options(xtflitemicro PRIVATE ${BUILD_FLAGS})
target_compile_definitions(xtflitemicro PUBLIC
    ${DEFINTIONS}
)

set(header_list "")
foreach(DIR ${ALL_INCLUDES})
    file(GLOB_RECURSE headers "${DIR}/*.h")
    list(APPEND header_list ${headers})
endforeach()

target_sources(xtflitemicro
    PRIVATE ${TFLM_KERNEL_SOURCES}
    PRIVATE ${TFLITE_SOURCES}
    PRIVATE ${NN_SOURCES}
    PRIVATE ${XTFLIB_KERNEL_SOURCES}
    PRIVATE ${XTFLIB_XCORE_SOURCES}
    PUBLIC FILE_SET HEADERS
    BASE_DIRS "${TOP_DIR}/.."
    FILES ${header_list}
)
target_include_directories(xtflitemicro
    PRIVATE ${ALL_INCLUDES}
)
install(TARGETS xtflitemicro DESTINATION "${CMAKE_BINARY_DIR}/lib"
    FILE_SET HEADERS DESTINATION "${CMAKE_BINARY_DIR}/include")

# Move headers to clean up include directory
add_custom_target(project-install DEPENDS xtflitemicro
    COMMAND "${CMAKE_COMMAND}" --build "${PROJECT_BINARY_DIR}" --target install
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/lib_tflite_micro/submodules/tflite-micro/tensorflow" "${CMAKE_BINARY_DIR}/include"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/lib_tflite_micro/submodules/flatbuffers/include/flatbuffers" "${CMAKE_BINARY_DIR}/include"
    COMMAND rm -rf "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/lib_tflite_micro/submodules"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/lib_tflite_micro" "${CMAKE_BINARY_DIR}/include/temp_ltm"
    COMMAND rm -rf "${CMAKE_BINARY_DIR}/include/lib_tflite_micro"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/temp_ltm" "${CMAKE_BINARY_DIR}/include/lib_tflite_micro"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_nn/lib_nn" "${CMAKE_BINARY_DIR}/include/temp_lnn"
    COMMAND rm -rf "${CMAKE_BINARY_DIR}/include/lib_nn"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/temp_lnn" "${CMAKE_BINARY_DIR}/include/lib_nn"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/api/flash_server.h" "${CMAKE_BINARY_DIR}/include"
    )

# Create zip of lib and headers
add_custom_target(create_zip DEPENDS project-install COMMAND
    ${CMAKE_COMMAND} -E tar "cfv" "release_archive.zip" --format=zip
    "${CMAKE_BINARY_DIR}/lib" "${CMAKE_BINARY_DIR}/include")