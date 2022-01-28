#include "xcore_utils.h"

#include <complex>

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

TfLiteStatus GetSizeOfType(TfLiteContext *context, const TfLiteType type,
                           size_t *bytes) {
  // TODO(levp): remove the default case so that new types produce compilation
  // error.
  switch (type) {
  case kTfLiteFloat32:
    *bytes = sizeof(float);
    break;
  case kTfLiteInt32:
    *bytes = sizeof(int);
    break;
  case kTfLiteUInt8:
    *bytes = sizeof(uint8_t);
    break;
  case kTfLiteInt64:
    *bytes = sizeof(int64_t);
    break;
  case kTfLiteUInt64:
    *bytes = sizeof(uint64_t);
    break;
  case kTfLiteBool:
    *bytes = sizeof(bool);
    break;
  case kTfLiteComplex64:
    *bytes = sizeof(std::complex<float>);
    break;
  case kTfLiteComplex128:
    *bytes = sizeof(std::complex<double>);
    break;
  case kTfLiteInt16:
    *bytes = sizeof(int16_t);
    break;
  case kTfLiteInt8:
    *bytes = sizeof(int8_t);
    break;
  case kTfLiteFloat16:
    *bytes = sizeof(TfLiteFloat16);
    break;
  case kTfLiteFloat64:
    *bytes = sizeof(double);
    break;
  default:
    if (context) {
      context->ReportError(
          context,
          "Type %d is unsupported. Only float16, float32, float64, int8, "
          "int16, int32, int64, uint8, uint64, bool, complex64 and "
          "complex128 supported currently.",
          type);
    }
    return kTfLiteError;
  }
  return kTfLiteOk;
}

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
} // namespace tflite
