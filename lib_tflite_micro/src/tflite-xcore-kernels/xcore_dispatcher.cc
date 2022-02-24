// Copyright (c) 2020, XMOS Ltd, All rights reserved
#include "xcore_dispatcher.h"

#include <cassert>

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

static Dispatcher *kDispatcher = nullptr;

void SetDispatcher(Dispatcher *dispatcher) { kDispatcher = dispatcher; }

Dispatcher *GetDispatcher() {
  assert(kDispatcher);
  return kDispatcher;
}

Dispatcher::Dispatcher(tflite::ErrorReporter *reporter, bool use_current_core)
    : use_current_thread_(use_current_core), reporter_(reporter) {
  tasks_.size = 0;
}

Dispatcher::~Dispatcher() {}

TfLiteStatus Dispatcher::JoinTasks() {
  if (tasks_.size == 0)
    return kTfLiteOk;

  // NOTE: use_current_thread_ is ignored on non-xcore targets
  int begin = 0;

  // Start threads
  for (int i = begin; i < tasks_.size; i++) {
    tasks_.function(tasks_.arguments[i]);
  }

  tasks_.size = 0;

  return kTfLiteOk;
}

tflite::ErrorReporter *Dispatcher::GetReporter() { return reporter_; }

TfLiteStatus Dispatcher::Reset() {
  tasks_.size = 0;

  return kTfLiteOk;
}

TfLiteStatus Dispatcher::InitializeTasks(thread_function_t function,
                                         char *stack, size_t stack_size) {
  tasks_.function = function;
  tasks_.stack_size = stack_size;
  tasks_.size = 0;
  tasks_.stack = stack;

  return kTfLiteOk;
}

TfLiteStatus Dispatcher::AddTask(void *argument) {
  assert(tasks_.size < kMaxThreads);

  if (tasks_.size < kMaxThreads) {
    tasks_.arguments[tasks_.size] = argument;
    tasks_.size++;

    return kTfLiteOk;
  }

  return kTfLiteError;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
