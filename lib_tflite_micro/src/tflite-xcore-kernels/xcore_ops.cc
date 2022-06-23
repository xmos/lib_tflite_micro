#include "xcore_ops.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void RegisterXCOps(MicroOpResolver *res){
  auto *resolver = reinterpret_cast<MicroMutableOpResolver<XTFLM_OPERATORS>*>(res);
  resolver->AddCustom(XC_conv2d_v2_OpCode,
                      Register_XC_conv2d_v2());
  resolver->AddCustom(tflite::ops::micro::xcore::XC_strided_slice_OpCode,
                      tflite::ops::micro::xcore::Register_XC_strided_slice());
  resolver->AddCustom(tflite::ops::micro::xcore::XC_ld_flash_OpCode,
                      tflite::ops::micro::xcore::Register_XC_ld_flash());
  resolver->AddCustom(tflite::ops::micro::xcore::XC_bsign_8_OpCode,
                      tflite::ops::micro::xcore::Register_XC_bsign_8());
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite