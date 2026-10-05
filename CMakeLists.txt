
cmake_minimum_required(VERSION 3.23)
project(lib_tflite_micro)
enable_language(C CXX ASM)

if("${CMAKE_CURRENT_LIST_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
	message(FATAL_ERROR "Use an out-of-source build (e.g. cmake -B build).")
endif()

add_subdirectory(lib_tflite_micro)
