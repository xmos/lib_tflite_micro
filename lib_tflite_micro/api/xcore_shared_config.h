// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_SHARED_CONFIG_H_
#define XCORE_SHARED_CONFIG_H_

namespace shared_config {

// This string is used as a key to store the shared config
// between xformer and lib_tflite_micro in the flatbuffer
constexpr char xcoreMetadataName[] = "xcoreSharedConfig";

struct xcore_metadata {
  // Dummy variable for aligning the metadata struct to 16 bytes
  // We cannot use alignas(16) yet in xcore
  int32_t padding[2];
  // Versions of libraries used to build the model
  int32_t lib_nn_major_version;
  int32_t lib_nn_minor_version;
  int32_t lib_nn_patch_version;
  int32_t lib_tflite_micro_major_version;
  int32_t lib_tflite_micro_minor_version;
  int32_t lib_tflite_micro_patch_version;
  int32_t xformer_major_version;
  int32_t xformer_minor_version;
  int32_t xformer_patch_version;
  // Number of threads required from the runtime to execute the model
  int32_t required_thread_count;
};

} // namespace shared_config

#endif // XCORE_SHARED_CONFIG_H_