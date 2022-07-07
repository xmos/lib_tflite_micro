#ifndef XCORE_CONFIG_H_
#define XCORE_CONFIG_H_

#include "../thread_call.h"

namespace tflite {
namespace micro {
namespace xcore {

struct xc_context_config_t {
  thread_info_t thread_info;
  void *flash_data; // channel to flash reader.
};

} // namespace xcore
} // namespace micro
} // namespace tflite

#endif // XCORE_CONFIG_H_
