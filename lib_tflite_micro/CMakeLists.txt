set(LIB_NAME "xtflitemicro" CACHE STRING "Name of the library to be built")
set(LIB_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/../lib")

# Remove MSVC Debug runtime checks before configuring lib_nn, which uses /O2.
if(MSVC)
    string(REPLACE "/RTC1" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

# Targets
include("${CMAKE_CURRENT_LIST_DIR}/sources.cmake")

add_library(${LIB_NAME} STATIC)
set_target_properties(${LIB_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${LIB_OUTPUT_DIR}")

target_sources(${LIB_NAME}
    PRIVATE ${TFLM_KERNEL_SOURCES}
    PRIVATE ${TFLITE_SOURCES}
    PRIVATE ${XTFLIB_KERNEL_SOURCES}
    PRIVATE $<TARGET_OBJECTS:lib_nn>
)

set(DEFINITIONS
    "__xtflm_conf_h_exists__"
    "NO_INTERPRETER"
    "TF_LITE_STATIC_MEMORY"
    "TF_LITE_DISABLE_X86_NEON"
    "TF_LITE_STRIP_ERROR_STRINGS"
)

# xs3 xcore
if(CMAKE_SYSTEM_PROCESSOR STREQUAL XCORE_XS3A)
    target_sources(${LIB_NAME} PRIVATE ${XTFLIB_XCORE_SOURCES})
    set_target_properties(${LIB_NAME} PROPERTIES OUTPUT_NAME "${LIB_NAME}_xs3a")
    set(BUILD_FLAGS
        "-O3"
        "-g"
        "-mcmodel=large"
        "-Wno-xcore-fptrgroup"
        "-Wno-bidirectional-buffered-port"
    )
# Riscv xcore
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL XCORE_VX4)
    target_sources(${LIB_NAME} PRIVATE ${XTFLIB_XCORE_SOURCES})
    set_target_properties(${LIB_NAME} PROPERTIES OUTPUT_NAME "${LIB_NAME}_vx4b")
    set(BUILD_FLAGS
        "-O3"
        "-g"
        "-Wno-fptrgroup"
        "-ffunction-sections"
        "-fdata-sections"
        "-Wl,--gc-sections"
    )
# MSVC native
elseif(MSVC)
    set(BUILD_FLAGS "/O2" "/MP16")
    list(APPEND DEFINITIONS "NN_USE_REF" "FLATBUFFERS_LOCALE_INDEPENDENT=0")
    target_compile_features(${LIB_NAME} PUBLIC cxx_std_17)
# Apple native
elseif(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET 12.0)
    set(BUILD_FLAGS "-O3" "-g")
    list(APPEND DEFINITIONS "NN_USE_REF" "FLATBUFFERS_LOCALE_INDEPENDENT=0")
    target_compile_features(${LIB_NAME} PUBLIC cxx_std_17)
# Native Others
else()
    set(BUILD_FLAGS "-O3" "-g")
    list(APPEND DEFINITIONS "NN_USE_REF" "FLATBUFFERS_LOCALE_INDEPENDENT=0")
    target_compile_features(${LIB_NAME} PUBLIC cxx_std_17)
endif()

# Link and compile options
target_link_libraries(${LIB_NAME} PUBLIC lib_nn)
target_compile_options(${LIB_NAME} PRIVATE ${BUILD_FLAGS})
target_compile_definitions(${LIB_NAME} PUBLIC ${DEFINITIONS})
target_include_directories(${LIB_NAME} PRIVATE ${ALL_INCLUDES})

# Project install
install(TARGETS ${LIB_NAME} ARCHIVE DESTINATION "${LIB_OUTPUT_DIR}")
set(INCLUDE_DIR "${CMAKE_BINARY_DIR}/include")

# lib_tflite_micro install
install(DIRECTORY
    "${TFLIB_DIR}/src"
    "${TFLIB_DIR}/api"
    DESTINATION "${INCLUDE_DIR}/lib_tflite_micro"
    COMPONENT headers
    FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY
    "${TFLIB_DIR}/submodules/tflite-micro/tensorflow"
    "${TFLIB_DIR}/submodules/tflite-micro/signal"
    "${TFLIB_DIR}/submodules/flatbuffers/include/flatbuffers"
    DESTINATION "${INCLUDE_DIR}"
    COMPONENT headers
    FILES_MATCHING PATTERN "*.h"
)
install(FILES
    "${TFLIB_DIR}/api/flash_server.h"
    "${TFLIB_DIR}/api/fast_flash.h"
    "${TFLIB_DIR}/api/ioserver.h"
    "${TFLIB_DIR}/api/tile_ram_server.h"
    DESTINATION "${INCLUDE_DIR}"
    COMPONENT headers
)

# lib_xud install
install(DIRECTORY "${LIB_XUD_ROOT_DIR}/lib_xud/api"
    DESTINATION "${INCLUDE_DIR}/lib_xud/lib_xud"
    COMPONENT headers
    FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY "${LIB_XUD_ROOT_DIR}/lib_xud/src/user"
    DESTINATION "${INCLUDE_DIR}/lib_xud/lib_xud/src"
    COMPONENT headers
    FILES_MATCHING PATTERN "*.h"
)

# lib_nn install
install(DIRECTORY "${LIB_NN_ROOT_DIR}/lib_nn"
    DESTINATION "${INCLUDE_DIR}"
    COMPONENT headers
    FILES_MATCHING PATTERN "*.h"
)

# Project install target
add_custom_target(project-install DEPENDS ${LIB_NAME}
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${INCLUDE_DIR}"
    COMMAND "${CMAKE_COMMAND}" --build "${PROJECT_BINARY_DIR}" --target install
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
