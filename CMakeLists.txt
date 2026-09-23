
# Project Definition
cmake_minimum_required(VERSION 3.23)
project(lib_tflite_micro)
enable_language(C CXX ASM)

set(LIB_NAME "xtflitemicro" CACHE STRING "Name of the library to be built")
set(LIB_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/lib")

# Disable in-source build
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(FATAL_ERROR "In-source build is not allowed! Please specify a build folder.\n\tex:cmake -B build")
endif()

# Targets
include("${CMAKE_CURRENT_SOURCE_DIR}/cmakefiles/xtflm.cmake")

add_library(${LIB_NAME} STATIC)
set_target_properties(${LIB_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${LIB_OUTPUT_DIR}")

set(header_list "")
foreach(DIR ${ALL_INCLUDES})
    file(GLOB_RECURSE headers "${DIR}/*.h")
    list(APPEND header_list ${headers})
endforeach()

target_sources(${LIB_NAME}
    PRIVATE ${TFLM_KERNEL_SOURCES}
    PRIVATE ${TFLITE_SOURCES}
    PRIVATE ${XTFLIB_KERNEL_SOURCES}
    PUBLIC FILE_SET HEADERS
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/.."
    FILES ${header_list}
)

set(DEFINITIONS
    "__xtflm_conf_h_exists__"
    "NO_INTERPRETER"
    "TF_LITE_STATIC_MEMORY"
    "TF_LITE_DISABLE_X86_NEON"
    "TF_LITE_STRIP_ERROR_STRINGS"
)

# Build flags
set(BUILD_FLAGS "-g")

# XS3
if(CMAKE_SYSTEM_PROCESSOR STREQUAL XCORE_XS3A)
    target_sources(${LIB_NAME} PRIVATE ${XTFLIB_XCORE_SOURCES})
    set_target_properties(${LIB_NAME} PROPERTIES OUTPUT_NAME "${LIB_NAME}_xs3a")
    list(APPEND BUILD_FLAGS
        "-mcmodel=large"
        "-Wno-xcore-fptrgroup"
        "-Wno-bidirectional-buffered-port"
    )
# VX4
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL XCORE_VX4)
    target_sources(${LIB_NAME} PRIVATE ${XTFLIB_XCORE_SOURCES})
    set_target_properties(${LIB_NAME} PROPERTIES OUTPUT_NAME "${LIB_NAME}_vx4b")    
    list(APPEND BUILD_FLAGS 
        "-Wno-fptrgroup"
        "-ffunction-sections"
        "-fdata-sections"
        "-Wl,--gc-sections"
    )
# Native
else()
    set(CMAKE_OSX_DEPLOYMENT_TARGET 12.0)
    list(APPEND DEFINITIONS "NN_USE_REF" "FLATBUFFERS_LOCALE_INDEPENDENT=0")
    target_compile_features(${LIB_NAME} PUBLIC cxx_std_17)
endif()

# Debug
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND BUILD_FLAGS "-00")
else()
    list(APPEND BUILD_FLAGS "-O3")
endif()

# Link and compile options
target_link_libraries(${LIB_NAME} PUBLIC lib_nn)
target_compile_options(${LIB_NAME} PRIVATE ${BUILD_FLAGS})
target_link_options(${LIB_NAME} PRIVATE ${BUILD_FLAGS})
target_compile_definitions(${LIB_NAME} PUBLIC ${DEFINITIONS})
target_include_directories(${LIB_NAME} PRIVATE ${ALL_INCLUDES})

# Project install
install(TARGETS ${LIB_NAME} 
    ARCHIVE DESTINATION "${LIB_OUTPUT_DIR}"
    FILE_SET HEADERS DESTINATION "${CMAKE_BINARY_DIR}/include"
)

# Move headers to clean up include directory
add_custom_target(project-install DEPENDS ${LIB_NAME}
    COMMAND rm -rf "${CMAKE_BINARY_DIR}/include"
    COMMAND "${CMAKE_COMMAND}" --build "${PROJECT_BINARY_DIR}" --target install
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/lib_tflite_micro/submodules/tflite-micro/tensorflow" "${CMAKE_BINARY_DIR}/include"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/lib_tflite_micro/submodules/tflite-micro/signal" "${CMAKE_BINARY_DIR}/include"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/lib_tflite_micro/submodules/flatbuffers/include/flatbuffers" "${CMAKE_BINARY_DIR}/include"
    COMMAND rm -rf "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/lib_tflite_micro/submodules"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/lib_tflite_micro" "${CMAKE_BINARY_DIR}/include/temp_ltm"
    COMMAND rm -rf "${CMAKE_BINARY_DIR}/include/lib_tflite_micro"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/temp_ltm" "${CMAKE_BINARY_DIR}/include/lib_tflite_micro"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_nn/lib_nn" "${CMAKE_BINARY_DIR}/include/temp_lnn"
    COMMAND rm -rf "${CMAKE_BINARY_DIR}/include/lib_nn"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/temp_lnn" "${CMAKE_BINARY_DIR}/include/lib_nn"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/api/flash_server.h" "${CMAKE_BINARY_DIR}/include"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/api/fast_flash.h" "${CMAKE_BINARY_DIR}/include"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/api/ioserver.h" "${CMAKE_BINARY_DIR}/include"
    COMMAND mv "${CMAKE_BINARY_DIR}/include/lib_tflite_micro/api/tile_ram_server.h" "${CMAKE_BINARY_DIR}/include"
    )

# Create zip of lib and headers
set(PACKAGE_DIR "${CMAKE_BINARY_DIR}/package")
add_custom_target(create_zip
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${PACKAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PACKAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${LIB_OUTPUT_DIR}" "${PACKAGE_DIR}/lib"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_BINARY_DIR}/include" "${PACKAGE_DIR}/include"
    COMMAND ${CMAKE_COMMAND} -E chdir "${PACKAGE_DIR}" ${CMAKE_COMMAND} -E tar "cfv" "${CMAKE_BINARY_DIR}/release_archive.zip" --format=zip lib include
    DEPENDS project-install)
