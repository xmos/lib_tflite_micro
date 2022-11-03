#ifndef XCORE_CONFIG_H_
#define XCORE_CONFIG_H_

#include "../src/thread_call.h"

#ifndef __XC__
struct xc_context_config_t {
  thread_info_t thread_info;
  void *flash_data; // channel to flash reader.
};
#else
struct xc_context_config_t {
  void * UNSAFE flash_data; // channel to flash reader.
};
#endif

#endif // XCORE_CONFIG_H_
