set(LIB_NN_REPOSITORY "https://github.com/xmos/lib_nn.git" CACHE STRING "lib_nn Git repository")
set(LIB_NN_TAG "develop" CACHE STRING "lib_nn Git tag or branch")

if(NOT IS_DIRECTORY "${LIB_NN_ROOT_DIR}")
  include(FetchContent)
  FetchContent_Declare(
    lib_nn
    GIT_REPOSITORY "${LIB_NN_REPOSITORY}"
    GIT_TAG "${LIB_NN_TAG}"
    SOURCE_DIR "${LIB_NN_ROOT_DIR}"
  )
  FetchContent_Populate(lib_nn)
endif()
