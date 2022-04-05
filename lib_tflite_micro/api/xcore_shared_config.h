// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_SHARED_CONFIG_H_
#define XCORE_SHARED_CONFIG_H_

namespace shared_config {

// This string is used as a key to store the shared config
// between xformer and lib_tflite_micro in the flatbuffer
constexpr char xcoreMetadataName[] = "xcoreSharedConfig";

struct xcore_metadata {
  // Versions of libraries used to build the model
  int lib_nn_major_version;
  int lib_nn_minor_version;
  int lib_nn_patch_version;
  int lib_tflite_micro_major_version;
  int lib_tflite_micro_minor_version;
  int lib_tflite_micro_patch_version;
  int xformer_major_version;
  int xformer_minor_version;
  int xformer_patch_version;
  // Number of threads required from the runtime to execute the model
  int required_thread_count;
} __attribute__((aligned (16)));

} // namespace shared_config

#endif // XCORE_SHARED_CONFIG_H_