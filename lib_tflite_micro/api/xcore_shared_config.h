// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_SHARED_CONFIG_H_
#define XCORE_SHARED_CONFIG_H_

namespace shared_config {

// This string is used as a key to store the shared config
// between xformer and lib_tflite_micro in the flatbuffer
constexpr char xcoreMetadataName[] = "xcoreSharedConfig";

constexpr int xcoreMaxNumOfTensors = 25;

struct tensor_info_t {
  uint32_t index;
  uint32_t external_address;
  uint32_t size;
};

// The metadata struct must be aligned to 16 bytes
// We cannot use alignas(16) yet in xcore
struct xcore_metadata_t {
  // Versions of libraries used to build the model
  uint32_t lib_nn_major_version;
  uint32_t lib_nn_minor_version;
  uint32_t lib_nn_patch_version;
  uint32_t lib_tflite_micro_major_version;
  uint32_t lib_tflite_micro_minor_version;
  uint32_t lib_tflite_micro_patch_version;
  uint32_t xformer_major_version;
  uint32_t xformer_minor_version;
  uint32_t xformer_patch_version;
  // Number of threads required from the runtime to execute the model
  uint32_t required_thread_count;
  // Number of input tensors loaded from external memory
  uint32_t num_external_input_tensors;
  // Number of output tensors loaded from external memory
  uint32_t num_external_output_tensors;
  tensor_info_t external_input_tensors_data[xcoreMaxNumOfTensors];
  tensor_info_t external_output_tensors_data[xcoreMaxNumOfTensors];
};

} // namespace shared_config

#endif // XCORE_SHARED_CONFIG_H_