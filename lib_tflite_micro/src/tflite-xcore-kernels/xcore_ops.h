#ifndef XCORE_OPS_H_
#define XCORE_OPS_H_

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

constexpr const char *XC_beta_activationf32_OpCode = "XC_beta_activationf32";
constexpr const char *XC_beta_concatf32_OpCode = "XC_beta_concatf32";
constexpr const char *XC_beta_convf32_OpCode = "XC_beta_convf32";
constexpr const char *XC_beta_transposeconvf32_OpCode = "XC_beta_transposeconvf32";
constexpr const char *XC_beta_fcf32_OpCode = "XC_beta_fcf32";

constexpr const char *XC_conv2d_v2_OpCode = "XC_conv2d_v2";
constexpr const char *XC_ld_flash_OpCode = "XC_ld_flash";
constexpr const char *XC_add_OpCode = "XC_add";
constexpr const char *XC_strided_slice_OpCode = "XC_strided_slice";
constexpr const char *XC_lookup_OpCode = "XC_lookup";
constexpr const char *XC_pad_OpCode = "XC_pad";
constexpr const char *XC_pad_3_to_4_OpCode = "XC_pad_3_to_4";
constexpr const char *XC_mul_OpCode = "XC_mul";
// Binarized ops
constexpr const char *XC_bsign_8_OpCode = "XC_bsign_8";

TfLiteRegistration_V1 *Register_XC_beta_activationf32();
TfLiteRegistration_V1 *Register_XC_beta_concatf32();
TfLiteRegistration_V1 *Register_XC_beta_convf32();
TfLiteRegistration_V1 *Register_XC_beta_transposeconvf32();
TfLiteRegistration_V1 *Register_XC_beta_fcf32();

TfLiteRegistration_V1 *Register_XC_conv2d_v2();
TfLiteRegistration_V1 *Register_XC_ld_flash();
TfLiteRegistration_V1 *Register_XC_add();
TfLiteRegistration_V1 *Register_XC_strided_slice();
TfLiteRegistration_V1 *Register_XC_lookup();
TfLiteRegistration_V1 *Register_XC_pad();
TfLiteRegistration_V1 *Register_XC_pad_3_to_4();
TfLiteRegistration_V1 *Register_XC_mul();
// Binarized ops
TfLiteRegistration_V1 *Register_XC_bsign_8();

void RegisterXCOps(tflite::MicroOpResolver *res);

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite

#endif // XCORE_OPS_H_
