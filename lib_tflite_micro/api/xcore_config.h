#ifndef XCORE_CONFIG_H_
#define XCORE_CONFIG_H_

#include "../src/thread_call.h"

struct xc_context_config_t {
  thread_info_t thread_info;
  void *flash_data; // channel to flash reader.
};

#endif // XCORE_CONFIG_H_
