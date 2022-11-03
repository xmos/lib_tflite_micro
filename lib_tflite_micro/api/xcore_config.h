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
  uint64_t id_aligned[2]; // Guarantee 64-bit alignment.        // Actual IDs           // ids of at most 4 threads - live during invoke
  uint32_t synchroniser;    // synchroniser for threads - live during invoke
  uint32_t nstackwords;     // nstackwords per stack   - live after load model
  void * UNSAFE stacks;             // pointer to top of stack - live after load model
  void * UNSAFE flash_data; // channel to flash reader.

};
#endif

#endif // XCORE_CONFIG_H_
