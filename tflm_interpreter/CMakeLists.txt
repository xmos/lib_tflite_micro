cmake_minimum_required(VERSION 3.14)

project(tflm_python VERSION 1.0.1)

# set host build
set(X86 ON)

# This variable is ignored on platforms other than Apple
set(CMAKE_OSX_SYSROOT /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk) 

set(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "C++ Compiler Base Flags" FORCE)


#**********************
# Build flags
#**********************

set(BUILD_FLAGS "-g -DTF_LITE_DISABLE_X86_NEON -D__tflm_conf_h_exists__ -O3 -DNN_USE_REF")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS ${BUILD_FLAGS})
set(CMAKE_C_FLAGS "${BUILD_FLAGS} -std=c99")

#**********************
# Build shared library
#**********************

add_library(tflm_python SHARED)
set_target_properties(tflm_python PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(tflm_python PROPERTIES PREFIX "")
target_link_libraries(tflm_python stdc++ m pthread)

include(../cmakefiles/tflm.cmake)

target_sources(tflm_python
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/dll_interpreter.cc"
  PRIVATE ${ALL_SOURCES}
)

target_include_directories(tflm_python
  PRIVATE ${ALL_INCLUDES}
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/tflm_interpreter/libs/linux")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/tflm_interpreter/libs/macos")
else() # must be Windows
  set(INSTALL_DIR "${PROJECT_SOURCE_DIR}/tflm_interpreter/libs/windows")
endif()

install(TARGETS tflm_python DESTINATION ${INSTALL_DIR})

