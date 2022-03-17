// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_DISPATCHER_H_
#define XCORE_DISPATCHER_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#ifdef XCORE
extern "C" {
#ifdef _TIME_H_
#define _clock_defined
#endif
#include "xcore/thread.h"
}

#define ATTRIBUTE_THREAD_FUNCTION __attribute__((fptrgroup("thread_function")))

#define STRINGIFY_THREAD_FUNCTION(NAME) #NAME
// #define GET_THREAD_FUNCTION_STACKWORDS(DEST, NAME)                     \
//   asm("ldc %[__dest], " STRINGIFY_THREAD_FUNCTION(NAME) ".nstackwords" \
//       : [ __dest ] "=r"(DEST))
#define GET_THREAD_FUNCTION_STACKSIZE(DEST, NAME)                              \
  {                                                                            \
    size_t _stack_words;                                                       \
    asm("ldc %[__dest], " STRINGIFY_THREAD_FUNCTION(NAME) ".nstackwords"       \
        : [__dest] "=r"(_stack_words));                                        \
    DEST = (_stack_words + 2) * 4;                                             \
  }

#else // not XCORE
#include <vector>

#define ATTRIBUTE_THREAD_FUNCTION
#define GET_THREAD_FUNCTION_STACKSIZE(DEST, NAME) DEST = 0

typedef void (*thread_function_t)(void *);
#endif

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

constexpr size_t kMaxThreads = 5;

typedef struct TaskArray {
  ATTRIBUTE_THREAD_FUNCTION thread_function_t function;
  size_t stack_size;
  char *stack;
  int size;
  void *arguments[kMaxThreads];
} TaskArray;

class Dispatcher {
public:
  Dispatcher(tflite::ErrorReporter *reporter, bool use_current_core = true);
  ~Dispatcher();

  TfLiteStatus InitializeTasks(thread_function_t function, char *stack,
                               size_t stack_size);
  TfLiteStatus AddTask(void *argument);
  TfLiteStatus JoinTasks();

  TfLiteStatus Reset();

  tflite::ErrorReporter *GetReporter();

private:
  bool use_current_thread_;
  TaskArray tasks_;
  tflite::ErrorReporter *reporter_;
};

// static, shared Dispatcher object
Dispatcher *GetDispatcher();
void SetDispatcher(Dispatcher *);

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite

#endif // XCORE_DISPATCHER_H_
