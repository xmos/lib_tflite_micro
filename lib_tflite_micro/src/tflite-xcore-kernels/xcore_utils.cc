#include "xcore_utils.h"

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {

size_t FetchBuffer(int8_t **dest, int8_t const *src, size_t size) {
  if (is_ram_address((uintptr_t)src)) {
    *dest = (int8_t *)src;
    return 0;
  } else {
    memload((void *)*dest, (void *)src, size);
    return size;
  }
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
