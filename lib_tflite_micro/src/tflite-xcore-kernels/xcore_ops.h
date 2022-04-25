#ifndef XCORE_OPS_H_
#define XCORE_OPS_H_

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "xcore_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

constexpr const char *Conv2D_V2_OpCode = "XC_conv2d_v2";
constexpr const char *Load_Flash_OpCode = "XC_ld_flash";
constexpr const char *Strided_Slice_OpCode = "XC_strided_slice";
constexpr const char *Strided_Slice_v2_OpCode = "XC_strided_slice_v2";
// Binarized ops
constexpr const char *Bsign_8_OpCode = "XC_bsign_8";

TfLiteRegistration *Register_Conv2D_V2();
TfLiteRegistration *Register_LoadFromFlash();
TfLiteRegistration *Register_Strided_Slice();
TfLiteRegistration *Register_Strided_Slice_V2();
// Binarized ops
TfLiteRegistration *Register_BSign_8();

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite

#endif // XCORE_OPS_H_
