// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_INTERPRETER_H_
#define XCORE_INTERPRETER_H_

#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/micro_memory_planner.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "xcore_profiler.h"

namespace tflite {
namespace micro {
namespace xcore {

class XCoreInterpreter : public tflite::MicroInterpreter {
public:
  XCoreInterpreter(const tflite::Model *model,
                   const tflite::MicroOpResolver &resolver,
                   tflite::MicroAllocator *allocator,
                   bool use_curent_thread = true,
                   XCoreProfiler *profiler = nullptr);

  static XCoreInterpreter *
  Create(uint8_t interpreter_buffer[], const tflite::Model *model,
         const tflite::MicroOpResolver &resolver, uint8_t *arena,
         size_t arena_size, bool use_current_thread, XCoreProfiler *profiler);

  void PrintMemoryPlan();
  TfLiteTensor *tensor(size_t tensor_index);
  const char *node_name(int sub_idx, int i);

  TfLiteStatus GetTensorDetails(size_t tensor_index, char *name, int name_len,
                                int *shape, int *type, float *scale,
                                int32_t *zero_point);

  TfLiteStatus GetTensorDetailsBufferSizes(size_t tensor_index, size_t *dims,
                                           size_t *scales, size_t *zero_points);

  size_t input_tensor_index(size_t input_index);
  size_t output_tensor_index(size_t output_index);
  const Model *model__;
  MicroAllocator *allocator_;
};

} // namespace xcore
} // namespace micro
} // namespace tflite

#endif // XCORE_INTERPRETER_H_
