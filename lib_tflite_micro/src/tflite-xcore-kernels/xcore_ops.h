#ifndef XCORE_OPS_H_
#define XCORE_OPS_H_

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {

constexpr const char *XC_beta_activationf32_OpCode = "XC_beta_activationf32";
constexpr const char *XC_beta_concatf32_OpCode = "XC_beta_concatf32";
constexpr const char *XC_beta_convf32_OpCode = "XC_beta_convf32";
constexpr const char *XC_beta_transposeconvf32_OpCode =
    "XC_beta_transposeconvf32";
constexpr const char *XC_beta_fcf32_OpCode = "XC_beta_fcf32";

constexpr const char *XC_binaryi16_OpCode = "XC_binaryi16";
constexpr const char *XC_unaryi16_OpCode = "XC_unaryi16";

constexpr const char *XC_conv2d_v2_OpCode = "XC_conv2d_v2";
constexpr const char *XC_maxpool2d_OpCode = "XC_maxpool2d";
constexpr const char *XC_softmax_OpCode = "XC_softmax";
constexpr const char *XC_batched_softmax_OpCode = "XC_batched_softmax";
constexpr const char *XC_ld_weights_OpCode = "XC_ld_weights";
constexpr const char *XC_add_OpCode = "XC_add";
constexpr const char *XC_slice_OpCode = "XC_slice";
constexpr const char *XC_broadcast_OpCode = "XC_broadcast";
constexpr const char *XC_lookup_OpCode = "XC_lookup";
constexpr const char *XC_pad_OpCode = "XC_pad";
constexpr const char *XC_concat_OpCode = "XC_concat";
constexpr const char *XC_pad_3_to_4_OpCode = "XC_pad_3_to_4";
constexpr const char *XC_mul_OpCode = "XC_mul";
constexpr const char *XC_mean_OpCode = "XC_mean";
constexpr const char *XC_expand_8_to_16_OpCode = "XC_expand_8_to_16";
// Binarized ops
constexpr const char *XC_bsign_8_OpCode = "XC_bsign_8";

TFLMRegistration *Register_XC_beta_activationf32();
TFLMRegistration *Register_XC_beta_concatf32();
TFLMRegistration *Register_XC_beta_convf32();
TFLMRegistration *Register_XC_beta_transposeconvf32();
TFLMRegistration *Register_XC_beta_fcf32();

TFLMRegistration *Register_XC_binaryi16();
TFLMRegistration *Register_XC_unaryi16();

TFLMRegistration *Register_XC_conv2d_v2();
TFLMRegistration *Register_XC_maxpool2d();
TFLMRegistration *Register_XC_softmax();
TFLMRegistration *Register_XC_batched_softmax();
TFLMRegistration *Register_XC_ld_weights();
TFLMRegistration *Register_XC_add();
TFLMRegistration *Register_XC_slice();
TFLMRegistration *Register_XC_broadcast();
TFLMRegistration *Register_XC_lookup();
TFLMRegistration *Register_XC_pad();
TFLMRegistration *Register_XC_concat();
TFLMRegistration *Register_XC_pad_3_to_4();
TFLMRegistration *Register_XC_mul();
TFLMRegistration *Register_XC_mean();
TFLMRegistration *Register_XC_expand_8_to_16();
// Binarized ops
TFLMRegistration *Register_XC_bsign_8();

void RegisterXCOps(tflite_micro::MicroOpResolver *res);

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro

#endif // XCORE_OPS_H_
