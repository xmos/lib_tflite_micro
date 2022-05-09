// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_SHARED_CONFIG_H_
#define XCORE_SHARED_CONFIG_H_

namespace shared_config {

// This string is used as a key to store the shared config
// between xformer and lib_tflite_micro in the flatbuffer
constexpr char xcoreMetadataName[] = "xcoreSharedConfig";

#ifdef XCORE
//C++ compiler on xcore only allows allignment of 8 or less
#define STRUCT_ALLIGNMENT alignas(8)
#else 
#define STRUCT_ALLIGNMENT alignas(16)
#endif

struct STRUCT_ALLIGNMENT xcore_metadata {
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