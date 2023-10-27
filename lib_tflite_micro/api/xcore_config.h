#ifndef XCORE_CONFIG_H_
#define XCORE_CONFIG_H_

#include "../src/thread_call.h"

struct xc_context_config_t {
  // This is the thread count specified in the compiler.
  // It's used by lookup op, beta float ops etc to split up work
  // in the Prepare phase.
  // Conv ops have their own thread count as the thread work is
  // calculated in the compiler.
  int model_thread_count;
  thread_info_t thread_info;
  void *UNSAFE flash_data; // channel to flash reader.
};

#endif // XCORE_CONFIG_H_
