#include "xcore_ops.h"

#if defined(__xtflm_conf_h_exists__)
#include "xtflm_conf.h"
#else
#ifndef XTFLM_OPERATORS
#define XTFLM_OPERATORS 10
#endif
#endif

#ifndef XCORE_TFLITE_MICRO_PATCHED
#error                                                                         \
    "tflite-micro patch not applied! Fix by running 'make patch' in lib_tflite_micro!"
#endif

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {

void RegisterXCOps(MicroOpResolver *res) {
  auto *resolver =
      reinterpret_cast<MicroMutableOpResolver<XTFLM_OPERATORS> *>(res);

  resolver->AddCustom(XC_beta_activationf32_OpCode,
                      Register_XC_beta_activationf32());
  resolver->AddCustom(XC_beta_concatf32_OpCode, Register_XC_beta_concatf32());
  resolver->AddCustom(XC_beta_convf32_OpCode, Register_XC_beta_convf32());
  resolver->AddCustom(XC_beta_transposeconvf32_OpCode,
                      Register_XC_beta_transposeconvf32());
  resolver->AddCustom(XC_beta_fcf32_OpCode, Register_XC_beta_fcf32());
  resolver->AddCustom(XC_binaryi16_OpCode, Register_XC_binaryi16());
  resolver->AddCustom(XC_unaryi16_OpCode, Register_XC_unaryi16());
  resolver->AddCustom(XC_conv2d_v2_OpCode, Register_XC_conv2d_v2());
  resolver->AddCustom(XC_maxpool2d_OpCode, Register_XC_maxpool2d());
  resolver->AddCustom(XC_softmax_OpCode, Register_XC_softmax());
  resolver->AddCustom(XC_batched_softmax_OpCode, Register_XC_batched_softmax());
  resolver->AddCustom(XC_add_OpCode, Register_XC_add());
  resolver->AddCustom(XC_slice_OpCode, Register_XC_slice());
  resolver->AddCustom(XC_broadcast_OpCode, Register_XC_broadcast());
  resolver->AddCustom(XC_ld_weights_OpCode, Register_XC_ld_weights());
  resolver->AddCustom(XC_bsign_8_OpCode, Register_XC_bsign_8());
  resolver->AddCustom(XC_lookup_OpCode, Register_XC_lookup());
  resolver->AddCustom(XC_pad_OpCode, Register_XC_pad());
  resolver->AddCustom(XC_concat_OpCode, Register_XC_concat());
  resolver->AddCustom(XC_pad_3_to_4_OpCode, Register_XC_pad_3_to_4());
  resolver->AddCustom(XC_mul_OpCode, Register_XC_mul());
  resolver->AddCustom(XC_mean_OpCode, Register_XC_mean());
  resolver->AddCustom(XC_expand_8_to_16_OpCode, Register_XC_expand_8_to_16());
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
