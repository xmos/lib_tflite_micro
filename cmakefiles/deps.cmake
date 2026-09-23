# Provides the lib_nn target and LIB_NN_ROOT_DIR for lib_nn/api includes.
include(FetchContent)

set(LIB_NN_REPOSITORY       "git@github.com:xalbertoisorna/lib_nn.git") #TODO change to upstream
set(LIB_NN_TAG              "fix/cmake")
set(LIB_XUD_REPOSITORY      "https://github.com/xmos/lib_xud.git")
set(LIB_XUD_TAG             "v2.4.0")

FetchContent_Declare(
	lib_nn
	GIT_REPOSITORY          "${LIB_NN_REPOSITORY}"
	GIT_TAG                 "${LIB_NN_TAG}"
	SOURCE_SUBDIR           lib_nn
)
FetchContent_MakeAvailable(lib_nn)
FetchContent_GetProperties(lib_nn SOURCE_DIR LIB_NN_ROOT_DIR)
target_include_directories(lib_nn INTERFACE "$<BUILD_INTERFACE:${LIB_NN_ROOT_DIR}>")

FetchContent_Declare(
	lib_xud
	GIT_REPOSITORY "${LIB_XUD_REPOSITORY}"
	GIT_TAG        "${LIB_XUD_TAG}"
)
FetchContent_Populate(lib_xud)
set(LIB_XUD_ROOT_DIR "${lib_xud_SOURCE_DIR}")
