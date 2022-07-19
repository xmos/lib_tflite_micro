#ifndef XCORE_OPS_H_
#define XCORE_OPS_H_

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

constexpr const char *XC_conv2d_v2_OpCode = "XC_conv2d_v2";
constexpr const char *XC_ld_flash_OpCode = "XC_ld_flash";
constexpr const char *XC_strided_slice_OpCode = "XC_strided_slice";
constexpr const char *XC_lookup_OpCode = "XC_lookup";
// Binarized ops
constexpr const char *XC_bsign_8_OpCode = "XC_bsign_8";

TfLiteRegistration *Register_XC_conv2d_v2();
TfLiteRegistration *Register_XC_ld_flash();
TfLiteRegistration *Register_XC_strided_slice();
TfLiteRegistration *Register_XC_lookup();
// Binarized ops
TfLiteRegistration *Register_XC_bsign_8();

void RegisterXCOps(tflite::MicroOpResolver *res);

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite

#endif // XCORE_OPS_H_
