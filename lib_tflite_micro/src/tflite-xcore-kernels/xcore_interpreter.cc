// Copyright (c) 2020, XMOS Ltd, All rights reserved

#include "xcore_interpreter.h"
#include "xcore_utils.h"
#include <iostream>

namespace tflite {
namespace micro {
namespace xcore {

XCoreInterpreter::XCoreInterpreter(const tflite::Model* model,
                                   const tflite::MicroOpResolver& resolver,
                                   tflite::MicroAllocator* allocator,
                                   tflite::ErrorReporter* reporter,
                                   bool use_current_thread,
                                   XCoreProfiler* profiler)
    : tflite::MicroInterpreter(model, resolver, allocator, reporter, profiler),
      dispatcher_(reporter, use_current_thread) {
  SetDispatcher(&dispatcher_);
  if (profiler) {
      profiler->Init(allocator, model->subgraphs()->Get(0)->operators()->size());
  }
}

XCoreInterpreter::XCoreInterpreter(const tflite::Model* model,
                                   const tflite::MicroOpResolver& resolver,
                                   uint8_t* arena, size_t arena_size,
                                   tflite::ErrorReporter* reporter,
                                   bool use_current_thread,
                                   XCoreProfiler* profiler)
    : XCoreInterpreter::XCoreInterpreter(
          model, resolver, MicroAllocator::Create(arena, arena_size, reporter),
          reporter, use_current_thread, profiler) {}

TfLiteTensor* XCoreInterpreter::tensor(size_t tensor_index) {
  auto ctx = context();
  return ctx.GetTensor(&ctx, tensor_index);
}

// This function retrieves a node's name
// It is slightly awkward as it needs to retrieve the graph - which is private
// to the micro-interpreter.
const char * XCoreInterpreter::node_name(int sub_idx, int i) {
    auto ctx = context();
    TfLiteIntArray* arg = NULL;
    ctx.GetExecutionPlan(&ctx, &arg);
    MicroGraph *graph = (MicroGraph *) arg;
    void * user_data = graph->GetAllocations()[sub_idx].node_and_registrations[i].node.user_data;

    if (user_data != NULL) {
        struct tflite::ops::micro::XCoreOpData * x = (struct tflite::ops::micro::XCoreOpData *) user_data;
        return x->name;
    }
    return NULL;
}

}  // namespace xcore
}  // namespace micro
}  // namespace tflite
