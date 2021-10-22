cmake_minimum_required(VERSION 3.14)

#**********************
# Setup XMOS toolchain
#**********************
#include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/xmos_toolchain.cmake")

project(model_runner VERSION 1.0.0)

enable_language(CXX C)

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

set(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "C++ Compiler Base Flags" FORCE)

set(BUILD_FLAGS
  "-O3"
  "-D__tflm_conf_h_exists__"
  "-DNN_USE_REF"
)

#**********************
# firmware targets
#**********************
add_executable(xcore_interpreter_cmdline)
target_compile_options(xcore_interpreter_cmdline PRIVATE ${BUILD_FLAGS})
target_link_options(xcore_interpreter_cmdline PRIVATE ${BUILD_FLAGS})

set(TFLIB_SRC_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../lib_tflite_micro/src/tflite-xcore-kernels")

set(TFLITE_SRC_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../lib_tflite_micro/submodules/tflite-micro/tensorflow/lite")

set(TFLM_SRC_DIR
  "${TFLITE_SRC_DIR}/micro")

set(NN_SRC_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../lib_nn/lib_nn/src")

file(GLOB_RECURSE LIB_NN_C_SOURCES "${NN_SRC_DIR}/c/*.c")

target_sources(xcore_interpreter_cmdline
  PRIVATE "${TFLITE_SRC_DIR}/c/common.c"

  ${LIB_NN_C_SOURCES}

  PRIVATE "${NN_SRC_DIR}/cpp/AggregateFn.cpp"
  PRIVATE "${NN_SRC_DIR}/cpp/AggregateFn_DW.cpp"
  PRIVATE "${NN_SRC_DIR}/cpp/Filter2d.cpp"
  PRIVATE "${NN_SRC_DIR}/cpp/MaxPool2d.cpp"
  PRIVATE "${NN_SRC_DIR}/cpp/MaxPoolAgg.cpp"
  PRIVATE "${NN_SRC_DIR}/cpp/MemCpyFn.cpp"
  PRIVATE "${NN_SRC_DIR}/cpp/OutputTransformFn.cpp"
  PRIVATE "${NN_SRC_DIR}/cpp/filt2d/conv2d_utils.cpp"
  PRIVATE "${NN_SRC_DIR}/cpp/filt2d/util.cpp"
  PRIVATE "${NN_SRC_DIR}/cpp/filt2d/geom/Filter2dGeometry.cpp"
  PRIVATE "${NN_SRC_DIR}/cpp/filt2d/geom/ImageGeometry.cpp"
  PRIVATE "${NN_SRC_DIR}/cpp/filt2d/geom/WindowGeometry.cpp"
  PRIVATE "${NN_SRC_DIR}/cpp/filt2d/geom/WindowLocation.cpp"
  PRIVATE "${NN_SRC_DIR}/asm/asm_constants.c"

  PRIVATE "${TFLITE_SRC_DIR}/core/api/error_reporter.cc"
  PRIVATE "${TFLITE_SRC_DIR}/core/api/tensor_utils.cc"
  PRIVATE "${TFLITE_SRC_DIR}/core/api/flatbuffer_conversions.cc"
  PRIVATE "${TFLITE_SRC_DIR}/core/api/op_resolver.cc"
  PRIVATE "${TFLITE_SRC_DIR}/kernels/kernel_util.cc"
  PRIVATE "${TFLITE_SRC_DIR}/kernels/internal/quantization_util.cc"
  PRIVATE "${TFLITE_SRC_DIR}/schema/schema_utils.cc"

  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../lib_tflite_micro/src/inference_engine.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_activations.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_add.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_arg_min_max.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_bconv2d.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_bsign.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_conv2d.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_conv2d_v2.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_custom_options.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_detection_post.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_dispatcher.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_error_reporter.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_extended_interpreter.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_fully_connected.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_interpreter.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_load_from_flash_v2.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_pad.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_planning.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_pooling.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_profiler.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_type_conversions.cc"
  PRIVATE "${TFLIB_SRC_DIR}/xcore_utils.cc"

  PRIVATE "${TFLM_SRC_DIR}/all_ops_resolver.cc"
  PRIVATE "${TFLM_SRC_DIR}/flatbuffer_utils.cc"
  PRIVATE "${TFLM_SRC_DIR}/memory_helpers.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_allocator.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_error_reporter.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_graph.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_interpreter.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_profiler.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_string.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_time.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_utils.cc"
  PRIVATE "${TFLM_SRC_DIR}/mock_micro_graph.cc"
  PRIVATE "${TFLM_SRC_DIR}/recording_micro_allocator.cc"
  PRIVATE "${TFLM_SRC_DIR}/recording_simple_memory_allocator.cc"
  PRIVATE "${TFLM_SRC_DIR}/simple_memory_allocator.cc"
  PRIVATE "${TFLM_SRC_DIR}/system_setup.cc"
  PRIVATE "${TFLM_SRC_DIR}/all_ops_resolver.cc"
  PRIVATE "${TFLM_SRC_DIR}/flatbuffer_utils.cc"
  PRIVATE "${TFLM_SRC_DIR}/memory_helpers.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_allocator.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_error_reporter.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_graph.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_interpreter.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_profiler.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_string.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_time.cc"
  PRIVATE "${TFLM_SRC_DIR}/micro_utils.cc"
  PRIVATE "${TFLM_SRC_DIR}/mock_micro_graph.cc"
  PRIVATE "${TFLM_SRC_DIR}/recording_micro_allocator.cc"
  PRIVATE "${TFLM_SRC_DIR}/recording_simple_memory_allocator.cc"
  PRIVATE "${TFLM_SRC_DIR}/simple_memory_allocator.cc"
  PRIVATE "${TFLM_SRC_DIR}/system_setup.cc"
  PRIVATE "${TFLM_SRC_DIR}/test_helpers.cc"
  PRIVATE "${TFLM_SRC_DIR}/memory_planner/greedy_memory_planner.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/activations.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/activations_common.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/add.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/add_n.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/arg_min_max.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/batch_to_space_nd.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/cast.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/ceil.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/circular_buffer.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/circular_buffer_flexbuffers_generated_data.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/comparisons.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/concatenation.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/conv.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/conv_common.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/cumsum.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/depth_to_space.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/depthwise_conv.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/depthwise_conv_common.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/dequantize.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/detection_postprocess_flexbuffers_generated_data.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/elementwise.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/elu.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/ethosu.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/exp.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/expand_dims.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/fill.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/floor.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/floor_div.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/floor_mod.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/fully_connected.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/fully_connected_common.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/gather.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/gather_nd.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/hard_swish.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/hard_swish_common.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/if.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/kernel_runner.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/kernel_util.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/l2_pool_2d.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/l2norm.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/leaky_relu.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/log_softmax.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/logical.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/logical_common.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/logistic.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/logistic_common.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/maximum_minimum.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/mul.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/neg.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/pack.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/pad.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/pooling.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/pooling_common.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/prelu.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/quantize.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/quantize_common.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/reduce.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/reshape.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/resize_bilinear.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/resize_nearest_neighbor.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/round.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/shape.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/softmax.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/softmax_common.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/space_to_batch_nd.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/space_to_depth.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/split.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/split_v.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/squeeze.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/strided_slice.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/sub.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/svdf.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/svdf_common.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/tanh.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/transpose.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/transpose_conv.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/unpack.cc"
  PRIVATE "${TFLM_SRC_DIR}/kernels/zeros_like.cc"

)

target_include_directories(xcore_interpreter_cmdline
  PRIVATE ${XCORE_INTERPRETER_INCLUDES}
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../lib_tflite_micro/api"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../lib_tflite_micro/src/tflite-xcore-kernels"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../lib_tflite_micro/submodules/tflite-micro"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../lib_tflite_micro/submodules/gemmlowp"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../lib_tflite_micro/submodules/ruy"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../lib_tflite_micro/submodules/flatbuffers/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../lib_nn/"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../lib_nn/lib_nn/api"
  PRIVATE "${XMOS_TOOL_PATH}/target/include/"
)

install(TARGETS xcore_interpreter_cmdline DESTINATION ${INSTALL_DIR})
