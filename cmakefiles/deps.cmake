# Provides the lib_nn target and LIB_NN_ROOT_DIR for lib_nn/api includes.
include(FetchContent)

set(LIB_NN_REPOSITORY       "https://github.com/xmos/lib_nn.git")
set(LIB_NN_TAG              "develop")
set(LIB_XUD_REPOSITORY      "https://github.com/xmos/lib_xud.git")
set(LIB_XUD_TAG             "v2.4.0")

FetchContent_Declare(
	lib_nn
	GIT_REPOSITORY          "${LIB_NN_REPOSITORY}"
	GIT_TAG                 "${LIB_NN_TAG}"
	SOURCE_DIR              "${CMAKE_CURRENT_LIST_DIR}/../../lib_nn"
)
FetchContent_Populate(lib_nn)
set(LIB_NN_ROOT_DIR "${lib_nn_SOURCE_DIR}")
add_subdirectory("${lib_nn_SOURCE_DIR}/lib_nn" "${lib_nn_BINARY_DIR}")
target_include_directories(lib_nn INTERFACE "$<BUILD_INTERFACE:${LIB_NN_ROOT_DIR}>")

FetchContent_Declare(
	lib_xud
	GIT_REPOSITORY "${LIB_XUD_REPOSITORY}"
	GIT_TAG        "${LIB_XUD_TAG}"
	SOURCE_DIR     "${CMAKE_CURRENT_LIST_DIR}/../../lib_xud"
)
FetchContent_Populate(lib_xud)
set(LIB_XUD_ROOT_DIR "${lib_xud_SOURCE_DIR}")
