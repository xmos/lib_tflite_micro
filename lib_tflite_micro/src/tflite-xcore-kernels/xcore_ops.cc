#include "xcore_ops.h"

#if defined(__xtflm_conf_h_exists__)
#include "xtflm_conf.h"
#else
#ifndef XTFLM_OPERATORS
#define XTFLM_OPERATORS 10
#endif
#endif

#ifndef XCORE_TFLITE_MICRO_PATCHED
#error "tflite-micro patch not applied! Fix by running 'make patch' in lib_tflite_micro!"
#endif

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void RegisterXCOps(MicroOpResolver *res) {
  auto *resolver =
      reinterpret_cast<MicroMutableOpResolver<XTFLM_OPERATORS> *>(res);

  resolver->AddCustom(XC_beta_convf32_OpCode, Register_XC_beta_convf32());
  resolver->AddCustom(XC_beta_transposeconvf32_OpCode, Register_XC_beta_transposeconvf32());
  resolver->AddCustom(XC_beta_fcf32_OpCode, Register_XC_beta_fcf32());

  resolver->AddCustom(XC_conv2d_v2_OpCode, Register_XC_conv2d_v2());
  resolver->AddCustom(tflite::ops::micro::xcore::XC_add_OpCode,
                      tflite::ops::micro::xcore::Register_XC_add());
  resolver->AddCustom(tflite::ops::micro::xcore::XC_strided_slice_OpCode,
                      tflite::ops::micro::xcore::Register_XC_strided_slice());
  resolver->AddCustom(tflite::ops::micro::xcore::XC_ld_flash_OpCode,
                      tflite::ops::micro::xcore::Register_XC_ld_flash());
  resolver->AddCustom(tflite::ops::micro::xcore::XC_bsign_8_OpCode,
                      tflite::ops::micro::xcore::Register_XC_bsign_8());
  resolver->AddCustom(tflite::ops::micro::xcore::XC_lookup_OpCode,
                      tflite::ops::micro::xcore::Register_XC_lookup());
  resolver->AddCustom(tflite::ops::micro::xcore::XC_pad_OpCode,
                      tflite::ops::micro::xcore::Register_XC_pad());
  resolver->AddCustom(tflite::ops::micro::xcore::XC_pad_3_to_4_OpCode,
                      tflite::ops::micro::xcore::Register_XC_pad_3_to_4());
  resolver->AddCustom(tflite::ops::micro::xcore::XC_mul_OpCode,
                      tflite::ops::micro::xcore::Register_XC_mul());
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite