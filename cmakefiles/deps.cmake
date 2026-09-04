# Provides the lib_nn target and LIB_NN_ROOT_DIR for lib_nn/api includes.
set(LIB_NN_REPOSITORY       "https://github.com/xmos/lib_nn.git")
set(LIB_NN_TAG              "develop")

include(FetchContent)
FetchContent_Declare(
	lib_nn
	GIT_REPOSITORY          "${LIB_NN_REPOSITORY}"
	GIT_TAG                 "${LIB_NN_TAG}"
	SOURCE_SUBDIR           lib_nn
)
FetchContent_MakeAvailable(lib_nn)
FetchContent_GetProperties(lib_nn SOURCE_DIR LIB_NN_ROOT_DIR)
target_include_directories(lib_nn INTERFACE "$<BUILD_INTERFACE:${LIB_NN_ROOT_DIR}>")
