// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_INTERPRETER_H_
#define XCORE_INTERPRETER_H_

#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/micro_memory_planner.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "xcore_dispatcher.h"
#include "xcore_profiler.h"
#include "../thread_call.h"

namespace tflite {
namespace micro {
namespace xcore {

class XCoreInterpreter : public tflite::MicroInterpreter {
public:
  XCoreInterpreter(const tflite::Model *model,
                   const tflite::MicroOpResolver &resolver,
                   tflite::MicroAllocator *allocator,
                   tflite::ErrorReporter *reporter,
                   tflite::GreedyMemoryPlanner *memory_planner__,
                   bool use_curent_thread = true,
                   XCoreProfiler *profiler = nullptr,
                   void *flash_data = nullptr);

  static XCoreInterpreter *
  Create(uint8_t interpreter_buffer[], const tflite::Model *model,
         const tflite::MicroOpResolver &resolver, uint8_t *arena,
         size_t arena_size, tflite::ErrorReporter *reporter,
         bool use_current_thread, XCoreProfiler *profiler, void *flash_data);

  void PrintMemoryPlan();
  TfLiteTensor *tensor(size_t tensor_index);
  const char *node_name(int sub_idx, int i);
  void *flash_data; // channel to flash reader.

  TfLiteStatus GetTensorDetails(size_t tensor_index, char *name, int name_len,
                                int *shape, int *type, float *scale,
                                int32_t *zero_point);

  TfLiteStatus GetTensorDetailsBufferSizes(size_t tensor_index, size_t *dims,
                                           size_t *scales, size_t *zero_points);

  size_t input_tensor_index(size_t input_index);
  size_t output_tensor_index(size_t output_index);
  const Model *model__;
  ErrorReporter* error_reporter__;
  tflite::GreedyMemoryPlanner* memory_planner__;
  thread_info_t thread_info;
 private:
  tflite::ops::micro::xcore::Dispatcher dispatcher_;
};

} // namespace xcore
} // namespace micro
} // namespace tflite

#endif // XCORE_INTERPRETER_H_
