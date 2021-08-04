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
                   XCoreProfiler* profiler = nullptr);

  XCoreInterpreter(const tflite::Model* model,
                   const tflite::MicroOpResolver& resolver,
                   tflite::MicroAllocator* allocator,
                   tflite::ErrorReporter* reporter,
                   bool use_current_thread = true,
                   XCoreProfiler* profiler = nullptr);

  TfLiteTensor* tensor(size_t tensor_index);

 private:
  tflite::ops::micro::xcore::Dispatcher dispatcher_;
};

}  // namespace xcore
}  // namespace micro
}  // namespace tflite

#endif  // XCORE_INTERPRETER_H_
