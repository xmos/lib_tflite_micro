// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_INTERPRETER_H_
#define XCORE_INTERPRETER_H_

#include "xcore_dispatcher.h"
#include "xcore_profiler.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

namespace tflite {
namespace micro {
namespace xcore {

class XCoreInterpreter : public tflite::MicroInterpreter {
 public:
  XCoreInterpreter(const tflite::Model* model,
                   const tflite::MicroOpResolver& resolver, uint8_t* arena,
                   size_t arena_size, tflite::ErrorReporter* reporter,
                   bool use_curent_thread = true,
                   XCoreProfiler* profiler = nullptr,
                   void *flash_data = nullptr);

  XCoreInterpreter(const tflite::Model* model,
                   const tflite::MicroOpResolver& resolver,
                   tflite::MicroAllocator* allocator,
                   tflite::ErrorReporter* reporter,
                   bool use_current_thread = true,
                   XCoreProfiler* profiler = nullptr,
                   void *flash_data = nullptr);

  TfLiteTensor* tensor(size_t tensor_index);
  const char *node_name(int sub_idx, int i);
  void *flash_data;  // channel to flash reader.

  TfLiteStatus GetTensorDetails(
    size_t tensor_index, char* name, int name_len, int* shape, int* type,
    float* scale, int32_t* zero_point);

  TfLiteStatus GetTensorDetailsBufferSizes(size_t tensor_index, size_t* dims,
                                           size_t* scales, size_t* zero_points);

  size_t input_tensor_index(size_t input_index);
  size_t output_tensor_index(size_t output_index);
  const Model *model__;
  ErrorReporter* error_reporter__;
 private:
  tflite::ops::micro::xcore::Dispatcher dispatcher_;
};

}  // namespace xcore
}  // namespace micro
}  // namespace tflite

#endif  // XCORE_INTERPRETER_H_
