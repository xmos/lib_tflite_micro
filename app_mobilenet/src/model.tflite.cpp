// This file is generated. Do not edit.
// Generated on: 28.02.2023 11:19:00


#include "../../api/xcore_config.h"
#include "lib_nn/api/version.h"
#include "lib_tflite_micro/api/version.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/micro_context.h"

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

// Check lib_nn and lib_tflite_micro versions
// NOTE: xformer version is saved for debugging purposes
// If lib_nn and lib_tflite_micro versions are as expected,
// then the xformer version doesn't matter as the model should execute
// If major version is zero, then minor versions must match
// Otherwise, major versions must match and binary minor version
// must be less or equal to runtime minor version
// Check if runtime lib_tflite_micro version matches with compiled version
static_assert((0 == 0 && lib_tflite_micro::major_version == 0 && 4 == lib_tflite_micro::minor_version) ||
              (0 == lib_tflite_micro::major_version) ||
              (4  < lib_tflite_micro::minor_version),
             "Model has been compiled with lib_tflite_micro version incompatible with runtime lib_tflite_micro version!");

// Check if runtime lib_nn version matches with compiled version
static_assert((0 == 0 && lib_nn::major_version == 0 && 1 == lib_nn::minor_version) ||
              (0 == lib_nn::major_version) ||
              (1  < lib_nn::minor_version),
             "Model has been compiled with lib_nn version incompatible with runtime lib_nn version!");

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
extern TfLiteRegistration *Register_XC_pad_3_to_4(void);
extern TfLiteRegistration *Register_XC_pad(void);
extern TfLiteRegistration *Register_XC_ld_flash(void);
extern TfLiteRegistration *Register_XC_conv2d_v2(void);
} // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

namespace {

constexpr int kTensorArenaSize = 414208;
uint8_t tensor_arena[kTensorArenaSize] ALIGN(8);
template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};
enum used_operators_e {
  OP_XC_pad_3_to_4, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_MEAN, OP_SOFTMAX,  OP_LAST
};

#ifdef TFLMC_XCORE_PROFILE
const char *op_strs[] = {
"OP_XC_pad_3_to_4", "OP_XC_pad", "OP_XC_ld_flash", "OP_XC_conv2d_v2", "OP_MEAN", "OP_SOFTMAX", };
int op_times[OP_LAST];
int op_counts[OP_LAST];
int64_t op_times_summed;
int time_t0, time_t1;
#endif

TfLiteContext ctx{};

TfLiteRegistration registrations[OP_LAST];
const TfArray<4, int> tensor_dimension0 = { 4, { 1,224,224,3 } };
const TfArray<1, float> quant0_scale = { 1, { 0.0039215688593685627, } };
const TfArray<1, int> quant0_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant0 = { (TfLiteFloatArray*)&quant0_scale, (TfLiteIntArray*)&quant0_zero, 0 };
const ALIGN(8) int32_t tensor_data1[2] = { 
    1, 2, 
};
const TfArray<1, int> tensor_dimension1 = { 1, { 2 } };
const ALIGN(8) int16_t tensor_data2[32] = { 
    7200, 23617, 9167, 0, 0, 2882, 3711, 4417, 6784, 4371, 
    6134, 9039, 2989, 0, 4768, 4356, -3011, -2069, -4841, -4097, 
    -4097, 121, -1585, -947, 2792, -125, -1851, -1138, -70, -4097, 
    -1297, -260, 
};
const TfArray<1, int> tensor_dimension2 = { 1, { 32 } };
const ALIGN(8) int16_t tensor_data3[24] = { 
    1005, 1235, 720, 2642, 1342, 69, 642, 27586, -83, -118, 
    -1658, 4631, -67, 219, 29, -31839, 0, 0, 0, 0, 
    0, 0, 0, 0, 
};
const TfArray<1, int> tensor_dimension3 = { 1, { 24 } };
const ALIGN(8) int16_t tensor_data4[24] = { 
    106, 379, 2, 0, 402, 25577, 138, 0, -76, -82, 
    -236, -257, 81, -341, 6, -257, 0, 0, 0, 0, 
    0, 0, 0, 0, 
};
const TfArray<1, int> tensor_dimension4 = { 1, { 24 } };
const TfArray<4, int> tensor_dimension5 = { 4, { 1,224,224,4 } };
const TfArray<1, float> quant5_scale = { 1, { 0.0039215688593685627, } };
const TfArray<1, int> quant5_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant5 = { (TfLiteFloatArray*)&quant5_scale, (TfLiteIntArray*)&quant5_zero, 0 };
const TfArray<4, int> tensor_dimension6 = { 4, { 1,225,225,4 } };
const TfArray<1, float> quant6_scale = { 1, { 0.0039215688593685627, } };
const TfArray<1, int> quant6_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant6 = { (TfLiteFloatArray*)&quant6_scale, (TfLiteIntArray*)&quant6_zero, 0 };
const TfArray<1, int> tensor_dimension7 = { 1, { 768 } };
const TfArray<4, int> tensor_dimension8 = { 4, { 1,112,112,8 } };
const TfArray<1, float> quant8_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant8_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant8 = { (TfLiteFloatArray*)&quant8_scale, (TfLiteIntArray*)&quant8_zero, 0 };
const TfArray<4, int> tensor_dimension9 = { 4, { 1,114,114,8 } };
const TfArray<1, float> quant9_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant9_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant9 = { (TfLiteFloatArray*)&quant9_scale, (TfLiteIntArray*)&quant9_zero, 0 };
const TfArray<1, int> tensor_dimension10 = { 1, { 160 } };
const TfArray<4, int> tensor_dimension11 = { 4, { 1,112,112,8 } };
const TfArray<1, float> quant11_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant11_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant11 = { (TfLiteFloatArray*)&quant11_scale, (TfLiteIntArray*)&quant11_zero, 0 };
const TfArray<4, int> tensor_dimension12 = { 4, { 1,112,112,8 } };
const TfArray<1, float> quant12_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant12_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant12 = { (TfLiteFloatArray*)&quant12_scale, (TfLiteIntArray*)&quant12_zero, 0 };
const TfArray<1, int> tensor_dimension13 = { 1, { 512 } };
const TfArray<4, int> tensor_dimension14 = { 4, { 1,112,112,16 } };
const TfArray<1, float> quant14_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant14_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant14 = { (TfLiteFloatArray*)&quant14_scale, (TfLiteIntArray*)&quant14_zero, 0 };
const TfArray<4, int> tensor_dimension15 = { 4, { 1,113,113,16 } };
const TfArray<1, float> quant15_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant15_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant15 = { (TfLiteFloatArray*)&quant15_scale, (TfLiteIntArray*)&quant15_zero, 0 };
const TfArray<1, int> tensor_dimension16 = { 1, { 160 } };
const TfArray<1, int> tensor_dimension17 = { 1, { 56 } };
const TfArray<4, int> tensor_dimension18 = { 4, { 1,56,56,16 } };
const TfArray<1, float> quant18_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant18_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant18 = { (TfLiteFloatArray*)&quant18_scale, (TfLiteIntArray*)&quant18_zero, 0 };
const TfArray<4, int> tensor_dimension19 = { 4, { 1,56,56,16 } };
const TfArray<1, float> quant19_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant19_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant19 = { (TfLiteFloatArray*)&quant19_scale, (TfLiteIntArray*)&quant19_zero, 0 };
const TfArray<1, int> tensor_dimension20 = { 1, { 768 } };
const TfArray<1, int> tensor_dimension21 = { 1, { 64 } };
const TfArray<4, int> tensor_dimension22 = { 4, { 1,56,56,32 } };
const TfArray<1, float> quant22_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant22_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant22 = { (TfLiteFloatArray*)&quant22_scale, (TfLiteIntArray*)&quant22_zero, 0 };
const TfArray<4, int> tensor_dimension23 = { 4, { 1,58,58,32 } };
const TfArray<1, float> quant23_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant23_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant23 = { (TfLiteFloatArray*)&quant23_scale, (TfLiteIntArray*)&quant23_zero, 0 };
const TfArray<1, int> tensor_dimension24 = { 1, { 304 } };
const TfArray<1, int> tensor_dimension25 = { 1, { 96 } };
const TfArray<4, int> tensor_dimension26 = { 4, { 1,56,56,32 } };
const TfArray<1, float> quant26_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant26_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant26 = { (TfLiteFloatArray*)&quant26_scale, (TfLiteIntArray*)&quant26_zero, 0 };
const TfArray<4, int> tensor_dimension27 = { 4, { 1,56,56,32 } };
const TfArray<1, float> quant27_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant27_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant27 = { (TfLiteFloatArray*)&quant27_scale, (TfLiteIntArray*)&quant27_zero, 0 };
const TfArray<1, int> tensor_dimension28 = { 1, { 1024 } };
const TfArray<1, int> tensor_dimension29 = { 1, { 64 } };
const TfArray<4, int> tensor_dimension30 = { 4, { 1,56,56,32 } };
const TfArray<1, float> quant30_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant30_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant30 = { (TfLiteFloatArray*)&quant30_scale, (TfLiteIntArray*)&quant30_zero, 0 };
const TfArray<4, int> tensor_dimension31 = { 4, { 1,57,57,32 } };
const TfArray<1, float> quant31_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant31_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant31 = { (TfLiteFloatArray*)&quant31_scale, (TfLiteIntArray*)&quant31_zero, 0 };
const TfArray<1, int> tensor_dimension32 = { 1, { 304 } };
const TfArray<1, int> tensor_dimension33 = { 1, { 64 } };
const TfArray<4, int> tensor_dimension34 = { 4, { 1,28,28,32 } };
const TfArray<1, float> quant34_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant34_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant34 = { (TfLiteFloatArray*)&quant34_scale, (TfLiteIntArray*)&quant34_zero, 0 };
const TfArray<4, int> tensor_dimension35 = { 4, { 1,28,28,32 } };
const TfArray<1, float> quant35_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant35_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant35 = { (TfLiteFloatArray*)&quant35_scale, (TfLiteIntArray*)&quant35_zero, 0 };
const TfArray<1, int> tensor_dimension36 = { 1, { 2048 } };
const TfArray<1, int> tensor_dimension37 = { 1, { 128 } };
const TfArray<4, int> tensor_dimension38 = { 4, { 1,28,28,64 } };
const TfArray<1, float> quant38_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant38_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant38 = { (TfLiteFloatArray*)&quant38_scale, (TfLiteIntArray*)&quant38_zero, 0 };
const TfArray<4, int> tensor_dimension39 = { 4, { 1,30,30,64 } };
const TfArray<1, float> quant39_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant39_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant39 = { (TfLiteFloatArray*)&quant39_scale, (TfLiteIntArray*)&quant39_zero, 0 };
const TfArray<1, int> tensor_dimension40 = { 1, { 592 } };
const TfArray<1, int> tensor_dimension41 = { 1, { 128 } };
const TfArray<4, int> tensor_dimension42 = { 4, { 1,28,28,64 } };
const TfArray<1, float> quant42_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant42_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant42 = { (TfLiteFloatArray*)&quant42_scale, (TfLiteIntArray*)&quant42_zero, 0 };
const TfArray<4, int> tensor_dimension43 = { 4, { 1,28,28,64 } };
const TfArray<1, float> quant43_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant43_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant43 = { (TfLiteFloatArray*)&quant43_scale, (TfLiteIntArray*)&quant43_zero, 0 };
const TfArray<1, int> tensor_dimension44 = { 1, { 4096 } };
const TfArray<1, int> tensor_dimension45 = { 1, { 128 } };
const TfArray<4, int> tensor_dimension46 = { 4, { 1,28,28,64 } };
const TfArray<1, float> quant46_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant46_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant46 = { (TfLiteFloatArray*)&quant46_scale, (TfLiteIntArray*)&quant46_zero, 0 };
const TfArray<4, int> tensor_dimension47 = { 4, { 1,29,29,64 } };
const TfArray<1, float> quant47_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant47_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant47 = { (TfLiteFloatArray*)&quant47_scale, (TfLiteIntArray*)&quant47_zero, 0 };
const TfArray<1, int> tensor_dimension48 = { 1, { 592 } };
const TfArray<1, int> tensor_dimension49 = { 1, { 128 } };
const TfArray<4, int> tensor_dimension50 = { 4, { 1,14,14,64 } };
const TfArray<1, float> quant50_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant50_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant50 = { (TfLiteFloatArray*)&quant50_scale, (TfLiteIntArray*)&quant50_zero, 0 };
const TfArray<4, int> tensor_dimension51 = { 4, { 1,14,14,64 } };
const TfArray<1, float> quant51_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant51_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant51 = { (TfLiteFloatArray*)&quant51_scale, (TfLiteIntArray*)&quant51_zero, 0 };
const TfArray<1, int> tensor_dimension52 = { 1, { 8192 } };
const TfArray<1, int> tensor_dimension53 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension54 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant54_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant54_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant54 = { (TfLiteFloatArray*)&quant54_scale, (TfLiteIntArray*)&quant54_zero, 0 };
const TfArray<4, int> tensor_dimension55 = { 4, { 1,16,16,128 } };
const TfArray<1, float> quant55_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant55_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant55 = { (TfLiteFloatArray*)&quant55_scale, (TfLiteIntArray*)&quant55_zero, 0 };
const TfArray<1, int> tensor_dimension56 = { 1, { 1168 } };
const TfArray<1, int> tensor_dimension57 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension58 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant58_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant58_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant58 = { (TfLiteFloatArray*)&quant58_scale, (TfLiteIntArray*)&quant58_zero, 0 };
const TfArray<4, int> tensor_dimension59 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant59_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant59_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant59 = { (TfLiteFloatArray*)&quant59_scale, (TfLiteIntArray*)&quant59_zero, 0 };
const TfArray<1, int> tensor_dimension60 = { 1, { 16384 } };
const TfArray<1, int> tensor_dimension61 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension62 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant62_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant62_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant62 = { (TfLiteFloatArray*)&quant62_scale, (TfLiteIntArray*)&quant62_zero, 0 };
const TfArray<4, int> tensor_dimension63 = { 4, { 1,16,16,128 } };
const TfArray<1, float> quant63_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant63_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant63 = { (TfLiteFloatArray*)&quant63_scale, (TfLiteIntArray*)&quant63_zero, 0 };
const TfArray<1, int> tensor_dimension64 = { 1, { 1168 } };
const TfArray<1, int> tensor_dimension65 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension66 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant66_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant66_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant66 = { (TfLiteFloatArray*)&quant66_scale, (TfLiteIntArray*)&quant66_zero, 0 };
const TfArray<4, int> tensor_dimension67 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant67_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant67_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant67 = { (TfLiteFloatArray*)&quant67_scale, (TfLiteIntArray*)&quant67_zero, 0 };
const TfArray<1, int> tensor_dimension68 = { 1, { 16384 } };
const TfArray<1, int> tensor_dimension69 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension70 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant70_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant70_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant70 = { (TfLiteFloatArray*)&quant70_scale, (TfLiteIntArray*)&quant70_zero, 0 };
const TfArray<4, int> tensor_dimension71 = { 4, { 1,16,16,128 } };
const TfArray<1, float> quant71_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant71_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant71 = { (TfLiteFloatArray*)&quant71_scale, (TfLiteIntArray*)&quant71_zero, 0 };
const TfArray<1, int> tensor_dimension72 = { 1, { 1168 } };
const TfArray<1, int> tensor_dimension73 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension74 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant74_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant74_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant74 = { (TfLiteFloatArray*)&quant74_scale, (TfLiteIntArray*)&quant74_zero, 0 };
const TfArray<4, int> tensor_dimension75 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant75_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant75_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant75 = { (TfLiteFloatArray*)&quant75_scale, (TfLiteIntArray*)&quant75_zero, 0 };
const TfArray<1, int> tensor_dimension76 = { 1, { 16384 } };
const TfArray<1, int> tensor_dimension77 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension78 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant78_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant78_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant78 = { (TfLiteFloatArray*)&quant78_scale, (TfLiteIntArray*)&quant78_zero, 0 };
const TfArray<4, int> tensor_dimension79 = { 4, { 1,16,16,128 } };
const TfArray<1, float> quant79_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant79_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant79 = { (TfLiteFloatArray*)&quant79_scale, (TfLiteIntArray*)&quant79_zero, 0 };
const TfArray<1, int> tensor_dimension80 = { 1, { 1168 } };
const TfArray<1, int> tensor_dimension81 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension82 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant82_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant82_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant82 = { (TfLiteFloatArray*)&quant82_scale, (TfLiteIntArray*)&quant82_zero, 0 };
const TfArray<4, int> tensor_dimension83 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant83_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant83_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant83 = { (TfLiteFloatArray*)&quant83_scale, (TfLiteIntArray*)&quant83_zero, 0 };
const TfArray<1, int> tensor_dimension84 = { 1, { 16384 } };
const TfArray<1, int> tensor_dimension85 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension86 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant86_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant86_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant86 = { (TfLiteFloatArray*)&quant86_scale, (TfLiteIntArray*)&quant86_zero, 0 };
const TfArray<4, int> tensor_dimension87 = { 4, { 1,16,16,128 } };
const TfArray<1, float> quant87_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant87_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant87 = { (TfLiteFloatArray*)&quant87_scale, (TfLiteIntArray*)&quant87_zero, 0 };
const TfArray<1, int> tensor_dimension88 = { 1, { 1168 } };
const TfArray<1, int> tensor_dimension89 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension90 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant90_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant90_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant90 = { (TfLiteFloatArray*)&quant90_scale, (TfLiteIntArray*)&quant90_zero, 0 };
const TfArray<4, int> tensor_dimension91 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant91_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant91_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant91 = { (TfLiteFloatArray*)&quant91_scale, (TfLiteIntArray*)&quant91_zero, 0 };
const TfArray<1, int> tensor_dimension92 = { 1, { 16384 } };
const TfArray<1, int> tensor_dimension93 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension94 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant94_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant94_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant94 = { (TfLiteFloatArray*)&quant94_scale, (TfLiteIntArray*)&quant94_zero, 0 };
const TfArray<4, int> tensor_dimension95 = { 4, { 1,15,15,128 } };
const TfArray<1, float> quant95_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant95_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant95 = { (TfLiteFloatArray*)&quant95_scale, (TfLiteIntArray*)&quant95_zero, 0 };
const TfArray<1, int> tensor_dimension96 = { 1, { 1168 } };
const TfArray<1, int> tensor_dimension97 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension98 = { 4, { 1,7,7,128 } };
const TfArray<1, float> quant98_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant98_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant98 = { (TfLiteFloatArray*)&quant98_scale, (TfLiteIntArray*)&quant98_zero, 0 };
const TfArray<4, int> tensor_dimension99 = { 4, { 1,7,7,128 } };
const TfArray<1, float> quant99_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant99_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant99 = { (TfLiteFloatArray*)&quant99_scale, (TfLiteIntArray*)&quant99_zero, 0 };
const TfArray<1, int> tensor_dimension100 = { 1, { 32768 } };
const TfArray<1, int> tensor_dimension101 = { 1, { 512 } };
const TfArray<4, int> tensor_dimension102 = { 4, { 1,7,7,256 } };
const TfArray<1, float> quant102_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant102_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant102 = { (TfLiteFloatArray*)&quant102_scale, (TfLiteIntArray*)&quant102_zero, 0 };
const TfArray<4, int> tensor_dimension103 = { 4, { 1,9,9,256 } };
const TfArray<1, float> quant103_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant103_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant103 = { (TfLiteFloatArray*)&quant103_scale, (TfLiteIntArray*)&quant103_zero, 0 };
const TfArray<1, int> tensor_dimension104 = { 1, { 2320 } };
const TfArray<1, int> tensor_dimension105 = { 1, { 768 } };
const TfArray<4, int> tensor_dimension106 = { 4, { 1,7,7,256 } };
const TfArray<1, float> quant106_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant106_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant106 = { (TfLiteFloatArray*)&quant106_scale, (TfLiteIntArray*)&quant106_zero, 0 };
const TfArray<4, int> tensor_dimension107 = { 4, { 1,7,7,256 } };
const TfArray<1, float> quant107_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant107_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant107 = { (TfLiteFloatArray*)&quant107_scale, (TfLiteIntArray*)&quant107_zero, 0 };
const TfArray<1, int> tensor_dimension108 = { 1, { 65536 } };
const TfArray<1, int> tensor_dimension109 = { 1, { 512 } };
const TfArray<4, int> tensor_dimension110 = { 4, { 1,7,7,256 } };
const TfArray<1, float> quant110_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant110_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant110 = { (TfLiteFloatArray*)&quant110_scale, (TfLiteIntArray*)&quant110_zero, 0 };
const TfArray<4, int> tensor_dimension111 = { 4, { 1,1,1,256 } };
const TfArray<1, float> quant111_scale = { 1, { 0.022377394139766693, } };
const TfArray<1, int> quant111_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant111 = { (TfLiteFloatArray*)&quant111_scale, (TfLiteIntArray*)&quant111_zero, 0 };
const TfArray<4, int> tensor_dimension112 = { 4, { 1,1,1,256 } };
const TfArray<1, float> quant112_scale = { 1, { 0.022377394139766693, } };
const TfArray<1, int> quant112_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant112 = { (TfLiteFloatArray*)&quant112_scale, (TfLiteIntArray*)&quant112_zero, 0 };
const TfArray<1, int> tensor_dimension113 = { 1, { 256256 } };
const TfArray<1, int> tensor_dimension114 = { 1, { 2008 } };
const TfArray<2, int> tensor_dimension115 = { 2, { 1,1000 } };
const TfArray<1, float> quant115_scale = { 1, { 0.12677012383937836, } };
const TfArray<1, int> quant115_zero = { 1, { -31 } };
const TfLiteAffineQuantization quant115 = { (TfLiteFloatArray*)&quant115_scale, (TfLiteIntArray*)&quant115_zero, 0 };
const TfArray<2, int> tensor_dimension116 = { 2, { 1,1000 } };
const TfArray<1, float> quant116_scale = { 1, { 0.00390625, } };
const TfArray<1, int> quant116_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant116 = { (TfLiteFloatArray*)&quant116_scale, (TfLiteIntArray*)&quant116_zero, 0 };
uint8_t ALIGN(4) opdata0[28] = { 112, 118, 0, 1, 4, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 128, 128, 128, 128, 6, 5, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs0 = { 1, { 0 } };
const TfArray<1, int> outputs0 = { 1, { 5 } };
uint8_t ALIGN(4) opdata1[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 224, 0, 0, 0, 0, 0, 0, 0, 128, 3, 0, 0, 4, 0, 0, 0, 132, 3, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs1 = { 1, { 5 } };
const TfArray<1, int> outputs1 = { 1, { 6 } };
uint8_t ALIGN(4) opdata2[45] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 1, 0, 0, 3, 5, 2, 18, 14, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 16, 83, 7, 0, 22, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs2 = 0; /* empty TfLiteIntArray */
const TfArray<1, int> outputs2 = { 1, { 7 } };
uint8_t ALIGN(4) opdata3[158] = { 107, 116, 0, 109, 112, 0, 40, 8, 7, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 252, 255, 255, 255, 228, 255, 255, 255, 120, 3, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 97, 103, 103, 112, 0, 8, 8, 0, 0, 0, 36, 0, 0, 0, 0, 111, 116, 112, 0, 8, 8, 0, 0, 0, 4, 0, 249, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 112, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 1, 34, 20, 7, 83, 43, 133, 131, 72, 59, 56, 7, 1, 7, 87, 14, 1, 137, 77, 0, 96, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs3 = { 3, { 6,7,4 } };
const TfArray<1, int> outputs3 = { 1, { 8 } };
uint8_t ALIGN(4) opdata4[61] = { 112, 112, 0, 24, 144, 3, 0, 0, 112, 0, 0, 0, 8, 0, 0, 0, 128, 3, 0, 0, 8, 0, 0, 0, 144, 3, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs4 = { 1, { 8 } };
const TfArray<1, int> outputs4 = { 1, { 9 } };
uint8_t ALIGN(4) opdata5[45] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 1, 0, 160, 0, 5, 2, 18, 14, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 82, 7, 0, 22, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs5 = 0; /* empty TfLiteIntArray */
const TfArray<1, int> outputs5 = { 1, { 10 } };
uint8_t ALIGN(4) opdata6[138] = { 107, 116, 0, 109, 112, 0, 8, 144, 3, 0, 0, 8, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 120, 3, 0, 0, 0, 111, 116, 112, 0, 8, 8, 0, 0, 0, 2, 0, 249, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 112, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 1, 34, 20, 7, 95, 43, 113, 111, 72, 59, 56, 7, 1, 7, 99, 14, 3, 117, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs6 = { 3, { 9,10,3 } };
const TfArray<1, int> outputs6 = { 1, { 11 } };
uint8_t ALIGN(4) opdata7[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 128, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs7 = { 1, { 11 } };
const TfArray<1, int> outputs7 = { 1, { 12 } };
uint8_t ALIGN(4) opdata8[45] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 1, 0, 0, 2, 5, 2, 18, 14, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 80, 7, 0, 22, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs8 = 0; /* empty TfLiteIntArray */
const TfArray<1, int> outputs8 = { 1, { 13 } };
uint8_t ALIGN(4) opdata9[158] = { 107, 116, 0, 109, 112, 0, 40, 128, 3, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 255, 255, 255, 232, 255, 255, 255, 120, 3, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 97, 103, 103, 112, 0, 8, 16, 0, 0, 0, 8, 0, 0, 0, 0, 111, 116, 112, 0, 8, 16, 0, 0, 0, 0, 0, 253, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 112, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 1, 34, 20, 7, 83, 43, 133, 131, 72, 59, 56, 7, 1, 7, 87, 14, 1, 137, 77, 0, 64, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs9 = { 3, { 12,13,2 } };
const TfArray<1, int> outputs9 = { 1, { 14 } };
uint8_t ALIGN(4) opdata10[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 16, 0, 0, 0, 16, 7, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs10 = { 1, { 14 } };
const TfArray<1, int> outputs10 = { 1, { 15 } };
uint8_t ALIGN(4) opdata11[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 160, 0, 112, 0, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 96, 79, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs11 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs11 = { 2, { 16,17 } };
uint8_t ALIGN(4) opdata12[138] = { 107, 116, 0, 109, 112, 0, 8, 32, 14, 0, 0, 32, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 224, 6, 0, 0, 0, 111, 116, 112, 0, 8, 16, 0, 0, 0, 248, 255, 0, 0, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 56, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 1, 34, 20, 7, 95, 43, 113, 111, 72, 59, 56, 7, 1, 7, 99, 14, 3, 117, 77, 1, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs12 = { 3, { 15,16,17 } };
const TfArray<1, int> outputs12 = { 1, { 18 } };
uint8_t ALIGN(4) opdata13[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 128, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs13 = { 1, { 18 } };
const TfArray<1, int> outputs13 = { 1, { 19 } };
uint8_t ALIGN(4) opdata14[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 3, 128, 0, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 224, 75, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs14 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs14 = { 2, { 20,21 } };
uint8_t ALIGN(4) opdata15[158] = { 107, 116, 0, 109, 112, 0, 40, 128, 3, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 255, 255, 255, 240, 255, 255, 255, 112, 3, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 97, 103, 103, 112, 0, 8, 32, 0, 0, 0, 16, 0, 0, 0, 0, 111, 116, 112, 0, 8, 32, 0, 0, 0, 1, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 56, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 1, 34, 20, 7, 83, 43, 133, 131, 72, 59, 56, 7, 1, 7, 87, 14, 1, 137, 77, 0, 64, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs15 = { 3, { 19,20,21 } };
const TfArray<1, int> outputs15 = { 1, { 22 } };
uint8_t ALIGN(4) opdata16[61] = { 112, 112, 0, 24, 64, 7, 0, 0, 56, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 32, 0, 0, 0, 64, 7, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs16 = { 1, { 22 } };
const TfArray<1, int> outputs16 = { 1, { 23 } };
uint8_t ALIGN(4) opdata17[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 48, 1, 192, 0, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 240, 73, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs17 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs17 = { 2, { 24,25 } };
uint8_t ALIGN(4) opdata18[138] = { 107, 116, 0, 109, 112, 0, 8, 64, 7, 0, 0, 32, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 32, 0, 0, 0, 224, 6, 0, 0, 0, 111, 116, 112, 0, 8, 32, 0, 0, 0, 249, 255, 0, 0, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 56, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 1, 34, 20, 7, 95, 43, 113, 111, 72, 59, 56, 7, 1, 7, 99, 14, 3, 117, 77, 1, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs18 = { 3, { 23,24,25 } };
const TfArray<1, int> outputs18 = { 1, { 26 } };
uint8_t ALIGN(4) opdata19[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs19 = { 1, { 26 } };
const TfArray<1, int> outputs19 = { 1, { 27 } };
uint8_t ALIGN(4) opdata20[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 4, 128, 0, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 69, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs20 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs20 = { 2, { 28,29 } };
uint8_t ALIGN(4) opdata21[142] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 32, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 224, 6, 0, 0, 0, 111, 116, 112, 0, 8, 32, 0, 0, 0, 2, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 56, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 1, 34, 20, 7, 99, 43, 117, 115, 72, 59, 56, 7, 1, 7, 103, 14, 0, 121, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs21 = { 3, { 27,28,29 } };
const TfArray<1, int> outputs21 = { 1, { 30 } };
uint8_t ALIGN(4) opdata22[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 32, 0, 0, 0, 32, 7, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs22 = { 1, { 30 } };
const TfArray<1, int> outputs22 = { 1, { 31 } };
uint8_t ALIGN(4) opdata23[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 48, 1, 128, 0, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 192, 67, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs23 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs23 = { 2, { 32,33 } };
uint8_t ALIGN(4) opdata24[138] = { 107, 116, 0, 109, 112, 0, 8, 64, 14, 0, 0, 64, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 32, 0, 0, 0, 192, 6, 0, 0, 0, 111, 116, 112, 0, 8, 32, 0, 0, 0, 2, 0, 251, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 1, 34, 20, 7, 95, 43, 113, 111, 72, 59, 56, 7, 1, 7, 99, 14, 3, 117, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs24 = { 3, { 31,32,33 } };
const TfArray<1, int> outputs24 = { 1, { 34 } };
uint8_t ALIGN(4) opdata25[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 128, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs25 = { 1, { 34 } };
const TfArray<1, int> outputs25 = { 1, { 35 } };
uint8_t ALIGN(4) opdata26[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 8, 0, 1, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 192, 58, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs26 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs26 = { 2, { 36,37 } };
uint8_t ALIGN(4) opdata27[142] = { 107, 116, 0, 109, 112, 0, 8, 128, 3, 0, 0, 32, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 96, 3, 0, 0, 0, 111, 116, 112, 0, 8, 64, 0, 0, 0, 3, 0, 251, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 1, 34, 20, 7, 99, 43, 117, 115, 72, 59, 56, 7, 1, 7, 103, 14, 0, 121, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs27 = { 3, { 35,36,37 } };
const TfArray<1, int> outputs27 = { 1, { 38 } };
uint8_t ALIGN(4) opdata28[61] = { 112, 112, 0, 24, 128, 7, 0, 0, 28, 0, 0, 0, 64, 0, 0, 0, 0, 7, 0, 0, 64, 0, 0, 0, 128, 7, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs28 = { 1, { 38 } };
const TfArray<1, int> outputs28 = { 1, { 39 } };
uint8_t ALIGN(4) opdata29[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 80, 2, 0, 1, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 55, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs29 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs29 = { 2, { 40,41 } };
uint8_t ALIGN(4) opdata30[138] = { 107, 116, 0, 109, 112, 0, 8, 128, 7, 0, 0, 64, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 64, 0, 0, 0, 192, 6, 0, 0, 0, 111, 116, 112, 0, 8, 64, 0, 0, 0, 1, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 1, 34, 20, 7, 95, 43, 113, 111, 72, 59, 56, 7, 1, 7, 99, 14, 3, 117, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs30 = { 3, { 39,40,41 } };
const TfArray<1, int> outputs30 = { 1, { 42 } };
uint8_t ALIGN(4) opdata31[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs31 = { 1, { 42 } };
const TfArray<1, int> outputs31 = { 1, { 43 } };
uint8_t ALIGN(4) opdata32[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 16, 0, 1, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 38, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs32 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs32 = { 2, { 44,45 } };
uint8_t ALIGN(4) opdata33[142] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 64, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 192, 6, 0, 0, 0, 111, 116, 112, 0, 8, 64, 0, 0, 0, 3, 0, 250, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 1, 34, 20, 7, 99, 43, 117, 115, 72, 59, 56, 7, 1, 7, 103, 14, 0, 121, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs33 = { 3, { 43,44,45 } };
const TfArray<1, int> outputs33 = { 1, { 46 } };
uint8_t ALIGN(4) opdata34[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 64, 0, 0, 0, 64, 7, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs34 = { 1, { 46 } };
const TfArray<1, int> outputs34 = { 1, { 47 } };
uint8_t ALIGN(4) opdata35[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 80, 2, 0, 1, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 32, 35, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs35 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs35 = { 2, { 48,49 } };
uint8_t ALIGN(4) opdata36[138] = { 107, 116, 0, 109, 112, 0, 8, 128, 14, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 64, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 64, 0, 0, 0, 2, 0, 250, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 1, 34, 20, 7, 95, 43, 113, 111, 72, 59, 56, 7, 1, 7, 99, 14, 3, 117, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs36 = { 3, { 47,48,49 } };
const TfArray<1, int> outputs36 = { 1, { 50 } };
uint8_t ALIGN(4) opdata37[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 128, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs37 = { 1, { 50 } };
const TfArray<1, int> outputs37 = { 1, { 51 } };
uint8_t ALIGN(4) opdata38[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 32, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 32, 1, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs38 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs38 = { 2, { 52,53 } };
uint8_t ALIGN(4) opdata39[142] = { 107, 116, 0, 109, 112, 0, 8, 128, 3, 0, 0, 64, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 64, 3, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 4, 0, 251, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 1, 34, 20, 7, 99, 43, 117, 115, 72, 59, 56, 7, 1, 7, 103, 14, 0, 121, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs39 = { 3, { 51,52,53 } };
const TfArray<1, int> outputs39 = { 1, { 54 } };
uint8_t ALIGN(4) opdata40[61] = { 112, 112, 0, 24, 0, 8, 0, 0, 14, 0, 0, 0, 128, 0, 0, 0, 0, 7, 0, 0, 128, 0, 0, 0, 0, 8, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs40 = { 1, { 54 } };
const TfArray<1, int> outputs40 = { 1, { 55 } };
uint8_t ALIGN(4) opdata41[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 144, 4, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 144, 250, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs41 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs41 = { 2, { 56,57 } };
uint8_t ALIGN(4) opdata42[138] = { 107, 116, 0, 109, 112, 0, 8, 0, 8, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 128, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 1, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 1, 34, 20, 7, 95, 43, 113, 111, 72, 59, 56, 7, 1, 7, 99, 14, 3, 117, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs42 = { 3, { 55,56,57 } };
const TfArray<1, int> outputs42 = { 1, { 58 } };
uint8_t ALIGN(4) opdata43[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs43 = { 1, { 58 } };
const TfArray<1, int> outputs43 = { 1, { 59 } };
uint8_t ALIGN(4) opdata44[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 64, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 144, 184, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs44 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs44 = { 2, { 60,61 } };
uint8_t ALIGN(4) opdata45[142] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 3, 0, 253, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 1, 34, 20, 7, 99, 43, 117, 115, 72, 59, 56, 7, 1, 7, 103, 14, 0, 121, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs45 = { 3, { 59,60,61 } };
const TfArray<1, int> outputs45 = { 1, { 62 } };
uint8_t ALIGN(4) opdata46[61] = { 112, 112, 0, 24, 0, 8, 0, 0, 14, 0, 0, 0, 128, 0, 0, 0, 0, 7, 0, 0, 128, 0, 0, 0, 0, 8, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs46 = { 1, { 62 } };
const TfArray<1, int> outputs46 = { 1, { 63 } };
uint8_t ALIGN(4) opdata47[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 144, 4, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 178, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs47 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs47 = { 2, { 64,65 } };
uint8_t ALIGN(4) opdata48[138] = { 107, 116, 0, 109, 112, 0, 8, 0, 8, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 128, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 2, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 1, 34, 20, 7, 95, 43, 113, 111, 72, 59, 56, 7, 1, 7, 99, 14, 3, 117, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs48 = { 3, { 63,64,65 } };
const TfArray<1, int> outputs48 = { 1, { 66 } };
uint8_t ALIGN(4) opdata49[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs49 = { 1, { 66 } };
const TfArray<1, int> outputs49 = { 1, { 67 } };
uint8_t ALIGN(4) opdata50[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 64, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 112, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs50 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs50 = { 2, { 68,69 } };
uint8_t ALIGN(4) opdata51[142] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 4, 0, 251, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 1, 34, 20, 7, 99, 43, 117, 115, 72, 59, 56, 7, 1, 7, 103, 14, 0, 121, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs51 = { 3, { 67,68,69 } };
const TfArray<1, int> outputs51 = { 1, { 70 } };
uint8_t ALIGN(4) opdata52[61] = { 112, 112, 0, 24, 0, 8, 0, 0, 14, 0, 0, 0, 128, 0, 0, 0, 0, 7, 0, 0, 128, 0, 0, 0, 0, 8, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs52 = { 1, { 70 } };
const TfArray<1, int> outputs52 = { 1, { 71 } };
uint8_t ALIGN(4) opdata53[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 144, 4, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 105, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs53 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs53 = { 2, { 72,73 } };
uint8_t ALIGN(4) opdata54[138] = { 107, 116, 0, 109, 112, 0, 8, 0, 8, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 128, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 2, 0, 250, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 1, 34, 20, 7, 95, 43, 113, 111, 72, 59, 56, 7, 1, 7, 99, 14, 3, 117, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs54 = { 3, { 71,72,73 } };
const TfArray<1, int> outputs54 = { 1, { 74 } };
uint8_t ALIGN(4) opdata55[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs55 = { 1, { 74 } };
const TfArray<1, int> outputs55 = { 1, { 75 } };
uint8_t ALIGN(4) opdata56[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 64, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 39, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs56 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs56 = { 2, { 76,77 } };
uint8_t ALIGN(4) opdata57[142] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 3, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 1, 34, 20, 7, 99, 43, 117, 115, 72, 59, 56, 7, 1, 7, 103, 14, 0, 121, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs57 = { 3, { 75,76,77 } };
const TfArray<1, int> outputs57 = { 1, { 78 } };
uint8_t ALIGN(4) opdata58[61] = { 112, 112, 0, 24, 0, 8, 0, 0, 14, 0, 0, 0, 128, 0, 0, 0, 0, 7, 0, 0, 128, 0, 0, 0, 0, 8, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs58 = { 1, { 78 } };
const TfArray<1, int> outputs58 = { 1, { 79 } };
uint8_t ALIGN(4) opdata59[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 144, 4, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 224, 32, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs59 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs59 = { 2, { 80,81 } };
uint8_t ALIGN(4) opdata60[138] = { 107, 116, 0, 109, 112, 0, 8, 0, 8, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 128, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 2, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 1, 34, 20, 7, 95, 43, 113, 111, 72, 59, 56, 7, 1, 7, 99, 14, 3, 117, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs60 = { 3, { 79,80,81 } };
const TfArray<1, int> outputs60 = { 1, { 82 } };
uint8_t ALIGN(4) opdata61[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs61 = { 1, { 82 } };
const TfArray<1, int> outputs61 = { 1, { 83 } };
uint8_t ALIGN(4) opdata62[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 64, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 224, 222, 5, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs62 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs62 = { 2, { 84,85 } };
uint8_t ALIGN(4) opdata63[142] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 3, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 1, 34, 20, 7, 99, 43, 117, 115, 72, 59, 56, 7, 1, 7, 103, 14, 0, 121, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs63 = { 3, { 83,84,85 } };
const TfArray<1, int> outputs63 = { 1, { 86 } };
uint8_t ALIGN(4) opdata64[61] = { 112, 112, 0, 24, 0, 8, 0, 0, 14, 0, 0, 0, 128, 0, 0, 0, 0, 7, 0, 0, 128, 0, 0, 0, 0, 8, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs64 = { 1, { 86 } };
const TfArray<1, int> outputs64 = { 1, { 87 } };
uint8_t ALIGN(4) opdata65[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 144, 4, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 80, 216, 5, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs65 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs65 = { 2, { 88,89 } };
uint8_t ALIGN(4) opdata66[138] = { 107, 116, 0, 109, 112, 0, 8, 0, 8, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 128, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 2, 0, 253, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 1, 34, 20, 7, 95, 43, 113, 111, 72, 59, 56, 7, 1, 7, 99, 14, 3, 117, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs66 = { 3, { 87,88,89 } };
const TfArray<1, int> outputs66 = { 1, { 90 } };
uint8_t ALIGN(4) opdata67[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs67 = { 1, { 90 } };
const TfArray<1, int> outputs67 = { 1, { 91 } };
uint8_t ALIGN(4) opdata68[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 64, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 80, 150, 5, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs68 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs68 = { 2, { 92,93 } };
uint8_t ALIGN(4) opdata69[142] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 4, 0, 250, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 1, 34, 20, 7, 99, 43, 117, 115, 72, 59, 56, 7, 1, 7, 103, 14, 0, 121, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs69 = { 3, { 91,92,93 } };
const TfArray<1, int> outputs69 = { 1, { 94 } };
uint8_t ALIGN(4) opdata70[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 128, 0, 0, 0, 128, 7, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs70 = { 1, { 94 } };
const TfArray<1, int> outputs70 = { 1, { 95 } };
uint8_t ALIGN(4) opdata71[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 144, 4, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 192, 143, 5, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs71 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs71 = { 2, { 96,97 } };
uint8_t ALIGN(4) opdata72[138] = { 107, 116, 0, 109, 112, 0, 8, 0, 15, 0, 0, 0, 1, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 128, 0, 0, 0, 0, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 2, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 1, 34, 20, 7, 95, 43, 113, 111, 72, 59, 56, 7, 1, 7, 99, 14, 3, 117, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs72 = { 3, { 95,96,97 } };
const TfArray<1, int> outputs72 = { 1, { 98 } };
uint8_t ALIGN(4) opdata73[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 128, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs73 = { 1, { 98 } };
const TfArray<1, int> outputs73 = { 1, { 99 } };
uint8_t ALIGN(4) opdata74[57] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 0, 0, 128, 0, 0, 0, 4, 0, 0, 6, 6, 2, 27, 23, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 192, 11, 5, 0, 32, 0, 0, 0, 6, 42, 10, 38, 1,  }; /* custom_initial_data */
const int inputs74 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs74 = { 2, { 100,101 } };
uint8_t ALIGN(4) opdata75[142] = { 107, 116, 0, 109, 112, 0, 8, 128, 3, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 111, 116, 112, 0, 8, 0, 1, 0, 0, 3, 0, 250, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 34, 20, 7, 99, 43, 117, 115, 72, 59, 56, 7, 1, 7, 103, 14, 0, 121, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs75 = { 3, { 99,100,101 } };
const TfArray<1, int> outputs75 = { 1, { 102 } };
uint8_t ALIGN(4) opdata76[61] = { 112, 112, 0, 24, 0, 9, 0, 0, 7, 0, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 1, 0, 0, 0, 9, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs76 = { 1, { 102 } };
const TfArray<1, int> outputs76 = { 1, { 103 } };
uint8_t ALIGN(4) opdata77[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 16, 9, 0, 6, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 176, 252, 4, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs77 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs77 = { 2, { 104,105 } };
uint8_t ALIGN(4) opdata78[138] = { 107, 116, 0, 109, 112, 0, 8, 0, 9, 0, 0, 0, 1, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 6, 0, 0, 0, 111, 116, 112, 0, 8, 0, 1, 0, 0, 249, 255, 0, 0, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 34, 20, 7, 95, 43, 113, 111, 72, 59, 56, 7, 1, 7, 99, 14, 3, 117, 77, 1, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs78 = { 3, { 103,104,105 } };
const TfArray<1, int> outputs78 = { 1, { 106 } };
uint8_t ALIGN(4) opdata79[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs79 = { 1, { 106 } };
const TfArray<1, int> outputs79 = { 1, { 107 } };
uint8_t ALIGN(4) opdata80[57] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0, 6, 6, 2, 27, 23, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 176, 248, 3, 0, 32, 0, 0, 0, 6, 42, 10, 38, 1,  }; /* custom_initial_data */
const int inputs80 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs80 = { 2, { 108,109 } };
uint8_t ALIGN(4) opdata81[142] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 0, 1, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 111, 116, 112, 0, 8, 0, 1, 0, 0, 3, 0, 249, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 34, 20, 7, 99, 43, 117, 115, 72, 59, 56, 7, 1, 7, 103, 14, 0, 121, 77, 0, 0, 20, 40, 4, 20, 20, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs81 = { 3, { 107,108,109 } };
const TfArray<1, int> outputs81 = { 1, { 110 } };
const TfLiteReducerParams opdata82 = { true };
const TfArray<2, int> inputs82 = { 2, { 110,1 } };
const TfArray<1, int> outputs82 = { 1, { 111 } };
uint8_t ALIGN(4) opdata83[61] = { 112, 112, 0, 24, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 112, 118, 0, 2, 33, 5, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 44, 0, 0, 0, 128, 128, 128, 128, 20, 6, 10, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs83 = { 1, { 111 } };
const TfArray<1, int> outputs83 = { 1, { 112 } };
uint8_t ALIGN(4) opdata84[39] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 0, 0, 233, 3, 0, 176, 15, 0, 0, 6, 6, 2, 27, 23, 2, 1, 2, 0, 17, 4, 42, 4, 36, 1,  }; /* custom_initial_data */
const int inputs84 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs84 = { 2, { 113,114 } };
uint8_t ALIGN(4) opdata85[168] = { 107, 116, 0, 109, 112, 0, 40, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 224, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 97, 103, 103, 112, 0, 8, 232, 3, 0, 0, 0, 1, 0, 0, 0, 111, 116, 112, 0, 8, 232, 3, 0, 0, 4, 0, 254, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 232, 3, 0, 0, 0, 1, 34, 20, 7, 83, 43, 133, 131, 72, 59, 56, 7, 0, 1, 0, 7, 0, 90, 0, 18, 0, 1, 0, 143, 0, 84, 0, 0, 0, 32, 1, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs85 = { 3, { 112,113,114 } };
const TfArray<1, int> outputs85 = { 1, { 115 } };
const TfLiteSoftmaxParams opdata86 = { 1 };
const TfArray<1, int> inputs86 = { 1, { 115 } };
const TfArray<1, int> outputs86 = { 1, { 116 } };
TfLiteTensor tflTensors[] = {
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension0, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant0)) }, {quant0.scale->data[0], quant0.zero_point->data[0] },150528, kTfLiteArenaRw, false, },
  { {(int32_t*)tensor_data1},(TfLiteIntArray*)&tensor_dimension1, kTfLiteInt32, {kTfLiteNoQuantization, nullptr }, {0,0},8, kTfLiteMmapRo, false, },
  { {(int32_t*)tensor_data2},(TfLiteIntArray*)&tensor_dimension2, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},64, kTfLiteMmapRo, false, },
  { {(int32_t*)tensor_data3},(TfLiteIntArray*)&tensor_dimension3, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},48, kTfLiteMmapRo, false, },
  { {(int32_t*)tensor_data4},(TfLiteIntArray*)&tensor_dimension4, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},48, kTfLiteMmapRo, false, },
  { {(int32_t*)(tensor_arena + 202512)},(TfLiteIntArray*)&tensor_dimension5, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant5)) }, {quant5.scale->data[0], quant5.zero_point->data[0] },200704, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension6, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant6)) }, {quant6.scale->data[0], quant6.zero_point->data[0] },202500, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 302864)},(TfLiteIntArray*)&tensor_dimension7, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},768, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 202512)},(TfLiteIntArray*)&tensor_dimension8, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant8)) }, {quant8.scale->data[0], quant8.zero_point->data[0] },100352, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension9, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant9)) }, {quant9.scale->data[0], quant9.zero_point->data[0] },103968, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 204320)},(TfLiteIntArray*)&tensor_dimension10, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},160, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 103968)},(TfLiteIntArray*)&tensor_dimension11, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant11)) }, {quant11.scale->data[0], quant11.zero_point->data[0] },100352, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension12, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant12)) }, {quant12.scale->data[0], quant12.zero_point->data[0] },100352, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 100352)},(TfLiteIntArray*)&tensor_dimension13, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 204304)},(TfLiteIntArray*)&tensor_dimension14, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant14)) }, {quant14.scale->data[0], quant14.zero_point->data[0] },200704, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension15, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant15)) }, {quant15.scale->data[0], quant15.zero_point->data[0] },204304, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 254480)},(TfLiteIntArray*)&tensor_dimension16, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},160, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 254640)},(TfLiteIntArray*)&tensor_dimension17, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},112, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 204304)},(TfLiteIntArray*)&tensor_dimension18, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant18)) }, {quant18.scale->data[0], quant18.zero_point->data[0] },50176, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension19, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant19)) }, {quant19.scale->data[0], quant19.zero_point->data[0] },50176, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension20, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},768, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50944)},(TfLiteIntArray*)&tensor_dimension21, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},128, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 107648)},(TfLiteIntArray*)&tensor_dimension22, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant22)) }, {quant22.scale->data[0], quant22.zero_point->data[0] },100352, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension23, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant23)) }, {quant23.scale->data[0], quant23.zero_point->data[0] },107648, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 208000)},(TfLiteIntArray*)&tensor_dimension24, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},304, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 208304)},(TfLiteIntArray*)&tensor_dimension25, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},192, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 107648)},(TfLiteIntArray*)&tensor_dimension26, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant26)) }, {quant26.scale->data[0], quant26.zero_point->data[0] },100352, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension27, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant27)) }, {quant27.scale->data[0], quant27.zero_point->data[0] },100352, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 100352)},(TfLiteIntArray*)&tensor_dimension28, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1024, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 101376)},(TfLiteIntArray*)&tensor_dimension29, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},128, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 103968)},(TfLiteIntArray*)&tensor_dimension30, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant30)) }, {quant30.scale->data[0], quant30.zero_point->data[0] },100352, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension31, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant31)) }, {quant31.scale->data[0], quant31.zero_point->data[0] },103968, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 129056)},(TfLiteIntArray*)&tensor_dimension32, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},304, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 129360)},(TfLiteIntArray*)&tensor_dimension33, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},128, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 103968)},(TfLiteIntArray*)&tensor_dimension34, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant34)) }, {quant34.scale->data[0], quant34.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension35, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant35)) }, {quant35.scale->data[0], quant35.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 25088)},(TfLiteIntArray*)&tensor_dimension36, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},2048, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 27136)},(TfLiteIntArray*)&tensor_dimension37, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},256, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 57600)},(TfLiteIntArray*)&tensor_dimension38, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant38)) }, {quant38.scale->data[0], quant38.zero_point->data[0] },50176, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension39, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant39)) }, {quant39.scale->data[0], quant39.zero_point->data[0] },57600, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 107776)},(TfLiteIntArray*)&tensor_dimension40, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},592, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 108368)},(TfLiteIntArray*)&tensor_dimension41, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},256, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 57600)},(TfLiteIntArray*)&tensor_dimension42, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant42)) }, {quant42.scale->data[0], quant42.zero_point->data[0] },50176, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension43, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant43)) }, {quant43.scale->data[0], quant43.zero_point->data[0] },50176, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 104000)},(TfLiteIntArray*)&tensor_dimension44, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},4096, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension45, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},256, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 53824)},(TfLiteIntArray*)&tensor_dimension46, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant46)) }, {quant46.scale->data[0], quant46.zero_point->data[0] },50176, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension47, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant47)) }, {quant47.scale->data[0], quant47.zero_point->data[0] },53824, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 66368)},(TfLiteIntArray*)&tensor_dimension48, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},592, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 66960)},(TfLiteIntArray*)&tensor_dimension49, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},256, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 53824)},(TfLiteIntArray*)&tensor_dimension50, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant50)) }, {quant50.scale->data[0], quant50.zero_point->data[0] },12544, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension51, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant51)) }, {quant51.scale->data[0], quant51.zero_point->data[0] },12544, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 12544)},(TfLiteIntArray*)&tensor_dimension52, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},8192, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 20736)},(TfLiteIntArray*)&tensor_dimension53, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&tensor_dimension54, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant54)) }, {quant54.scale->data[0], quant54.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension55, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant55)) }, {quant55.scale->data[0], quant55.zero_point->data[0] },32768, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 57856)},(TfLiteIntArray*)&tensor_dimension56, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1168, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 59024)},(TfLiteIntArray*)&tensor_dimension57, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant58)) }, {quant58.scale->data[0], quant58.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension59, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant59)) }, {quant59.scale->data[0], quant59.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 57856)},(TfLiteIntArray*)&tensor_dimension60, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},16384, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 25088)},(TfLiteIntArray*)&tensor_dimension61, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&tensor_dimension62, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant62)) }, {quant62.scale->data[0], quant62.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension63, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant63)) }, {quant63.scale->data[0], quant63.zero_point->data[0] },32768, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 57856)},(TfLiteIntArray*)&tensor_dimension64, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1168, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 59024)},(TfLiteIntArray*)&tensor_dimension65, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&tensor_dimension66, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant66)) }, {quant66.scale->data[0], quant66.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension67, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant67)) }, {quant67.scale->data[0], quant67.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 57856)},(TfLiteIntArray*)&tensor_dimension68, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},16384, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 25088)},(TfLiteIntArray*)&tensor_dimension69, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&tensor_dimension70, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant70)) }, {quant70.scale->data[0], quant70.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension71, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant71)) }, {quant71.scale->data[0], quant71.zero_point->data[0] },32768, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 57856)},(TfLiteIntArray*)&tensor_dimension72, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1168, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 59024)},(TfLiteIntArray*)&tensor_dimension73, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&tensor_dimension74, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant74)) }, {quant74.scale->data[0], quant74.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension75, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant75)) }, {quant75.scale->data[0], quant75.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 57856)},(TfLiteIntArray*)&tensor_dimension76, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},16384, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 25088)},(TfLiteIntArray*)&tensor_dimension77, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&tensor_dimension78, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant78)) }, {quant78.scale->data[0], quant78.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension79, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant79)) }, {quant79.scale->data[0], quant79.zero_point->data[0] },32768, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 57856)},(TfLiteIntArray*)&tensor_dimension80, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1168, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 59024)},(TfLiteIntArray*)&tensor_dimension81, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&tensor_dimension82, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant82)) }, {quant82.scale->data[0], quant82.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension83, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant83)) }, {quant83.scale->data[0], quant83.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 57856)},(TfLiteIntArray*)&tensor_dimension84, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},16384, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 25088)},(TfLiteIntArray*)&tensor_dimension85, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&tensor_dimension86, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant86)) }, {quant86.scale->data[0], quant86.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension87, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant87)) }, {quant87.scale->data[0], quant87.zero_point->data[0] },32768, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 57856)},(TfLiteIntArray*)&tensor_dimension88, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1168, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 59024)},(TfLiteIntArray*)&tensor_dimension89, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&tensor_dimension90, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant90)) }, {quant90.scale->data[0], quant90.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension91, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant91)) }, {quant91.scale->data[0], quant91.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 53888)},(TfLiteIntArray*)&tensor_dimension92, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},16384, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 25088)},(TfLiteIntArray*)&tensor_dimension93, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 28800)},(TfLiteIntArray*)&tensor_dimension94, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant94)) }, {quant94.scale->data[0], quant94.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension95, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant95)) }, {quant95.scale->data[0], quant95.zero_point->data[0] },28800, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 35072)},(TfLiteIntArray*)&tensor_dimension96, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1168, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 36240)},(TfLiteIntArray*)&tensor_dimension97, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 28800)},(TfLiteIntArray*)&tensor_dimension98, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant98)) }, {quant98.scale->data[0], quant98.zero_point->data[0] },6272, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 45312)},(TfLiteIntArray*)&tensor_dimension99, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant99)) }, {quant99.scale->data[0], quant99.zero_point->data[0] },6272, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension100, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},32768, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 51584)},(TfLiteIntArray*)&tensor_dimension101, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},1024, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&tensor_dimension102, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant102)) }, {quant102.scale->data[0], quant102.zero_point->data[0] },12544, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension103, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant103)) }, {quant103.scale->data[0], quant103.zero_point->data[0] },20736, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 33280)},(TfLiteIntArray*)&tensor_dimension104, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},2320, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 35600)},(TfLiteIntArray*)&tensor_dimension105, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},1536, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 20736)},(TfLiteIntArray*)&tensor_dimension106, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant106)) }, {quant106.scale->data[0], quant106.zero_point->data[0] },12544, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 78080)},(TfLiteIntArray*)&tensor_dimension107, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant107)) }, {quant107.scale->data[0], quant107.zero_point->data[0] },12544, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension108, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},65536, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 90624)},(TfLiteIntArray*)&tensor_dimension109, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},1024, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 65536)},(TfLiteIntArray*)&tensor_dimension110, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant110)) }, {quant110.scale->data[0], quant110.zero_point->data[0] },12544, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 1024)},(TfLiteIntArray*)&tensor_dimension111, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant111)) }, {quant111.scale->data[0], quant111.zero_point->data[0] },256, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 261568)},(TfLiteIntArray*)&tensor_dimension112, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant112)) }, {quant112.scale->data[0], quant112.zero_point->data[0] },256, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension113, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},256256, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 256256)},(TfLiteIntArray*)&tensor_dimension114, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},4016, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 260272)},(TfLiteIntArray*)&tensor_dimension115, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant115)) }, {quant115.scale->data[0], quant115.zero_point->data[0] },1000, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension116, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant116)) }, {quant116.scale->data[0], quant116.zero_point->data[0] },1000, kTfLiteArenaRw, false, },
};
TfLiteNode tflNodes[] = {
  { (TfLiteIntArray*)&inputs0, (TfLiteIntArray*)&outputs0, (TfLiteIntArray*)&inputs0, nullptr, const_cast<void*>(static_cast<const void*>(&opdata0)), nullptr, 28, },
  { (TfLiteIntArray*)&inputs1, (TfLiteIntArray*)&outputs1, (TfLiteIntArray*)&inputs1, nullptr, const_cast<void*>(static_cast<const void*>(&opdata1)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs2, (TfLiteIntArray*)&outputs2, (TfLiteIntArray*)&inputs2, nullptr, const_cast<void*>(static_cast<const void*>(&opdata2)), nullptr, 45, },
  { (TfLiteIntArray*)&inputs3, (TfLiteIntArray*)&outputs3, (TfLiteIntArray*)&inputs3, nullptr, const_cast<void*>(static_cast<const void*>(&opdata3)), nullptr, 158, },
  { (TfLiteIntArray*)&inputs4, (TfLiteIntArray*)&outputs4, (TfLiteIntArray*)&inputs4, nullptr, const_cast<void*>(static_cast<const void*>(&opdata4)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs5, (TfLiteIntArray*)&outputs5, (TfLiteIntArray*)&inputs5, nullptr, const_cast<void*>(static_cast<const void*>(&opdata5)), nullptr, 45, },
  { (TfLiteIntArray*)&inputs6, (TfLiteIntArray*)&outputs6, (TfLiteIntArray*)&inputs6, nullptr, const_cast<void*>(static_cast<const void*>(&opdata6)), nullptr, 138, },
  { (TfLiteIntArray*)&inputs7, (TfLiteIntArray*)&outputs7, (TfLiteIntArray*)&inputs7, nullptr, const_cast<void*>(static_cast<const void*>(&opdata7)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs8, (TfLiteIntArray*)&outputs8, (TfLiteIntArray*)&inputs8, nullptr, const_cast<void*>(static_cast<const void*>(&opdata8)), nullptr, 45, },
  { (TfLiteIntArray*)&inputs9, (TfLiteIntArray*)&outputs9, (TfLiteIntArray*)&inputs9, nullptr, const_cast<void*>(static_cast<const void*>(&opdata9)), nullptr, 158, },
  { (TfLiteIntArray*)&inputs10, (TfLiteIntArray*)&outputs10, (TfLiteIntArray*)&inputs10, nullptr, const_cast<void*>(static_cast<const void*>(&opdata10)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs11, (TfLiteIntArray*)&outputs11, (TfLiteIntArray*)&inputs11, nullptr, const_cast<void*>(static_cast<const void*>(&opdata11)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs12, (TfLiteIntArray*)&outputs12, (TfLiteIntArray*)&inputs12, nullptr, const_cast<void*>(static_cast<const void*>(&opdata12)), nullptr, 138, },
  { (TfLiteIntArray*)&inputs13, (TfLiteIntArray*)&outputs13, (TfLiteIntArray*)&inputs13, nullptr, const_cast<void*>(static_cast<const void*>(&opdata13)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs14, (TfLiteIntArray*)&outputs14, (TfLiteIntArray*)&inputs14, nullptr, const_cast<void*>(static_cast<const void*>(&opdata14)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs15, (TfLiteIntArray*)&outputs15, (TfLiteIntArray*)&inputs15, nullptr, const_cast<void*>(static_cast<const void*>(&opdata15)), nullptr, 158, },
  { (TfLiteIntArray*)&inputs16, (TfLiteIntArray*)&outputs16, (TfLiteIntArray*)&inputs16, nullptr, const_cast<void*>(static_cast<const void*>(&opdata16)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs17, (TfLiteIntArray*)&outputs17, (TfLiteIntArray*)&inputs17, nullptr, const_cast<void*>(static_cast<const void*>(&opdata17)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs18, (TfLiteIntArray*)&outputs18, (TfLiteIntArray*)&inputs18, nullptr, const_cast<void*>(static_cast<const void*>(&opdata18)), nullptr, 138, },
  { (TfLiteIntArray*)&inputs19, (TfLiteIntArray*)&outputs19, (TfLiteIntArray*)&inputs19, nullptr, const_cast<void*>(static_cast<const void*>(&opdata19)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs20, (TfLiteIntArray*)&outputs20, (TfLiteIntArray*)&inputs20, nullptr, const_cast<void*>(static_cast<const void*>(&opdata20)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs21, (TfLiteIntArray*)&outputs21, (TfLiteIntArray*)&inputs21, nullptr, const_cast<void*>(static_cast<const void*>(&opdata21)), nullptr, 142, },
  { (TfLiteIntArray*)&inputs22, (TfLiteIntArray*)&outputs22, (TfLiteIntArray*)&inputs22, nullptr, const_cast<void*>(static_cast<const void*>(&opdata22)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs23, (TfLiteIntArray*)&outputs23, (TfLiteIntArray*)&inputs23, nullptr, const_cast<void*>(static_cast<const void*>(&opdata23)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs24, (TfLiteIntArray*)&outputs24, (TfLiteIntArray*)&inputs24, nullptr, const_cast<void*>(static_cast<const void*>(&opdata24)), nullptr, 138, },
  { (TfLiteIntArray*)&inputs25, (TfLiteIntArray*)&outputs25, (TfLiteIntArray*)&inputs25, nullptr, const_cast<void*>(static_cast<const void*>(&opdata25)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs26, (TfLiteIntArray*)&outputs26, (TfLiteIntArray*)&inputs26, nullptr, const_cast<void*>(static_cast<const void*>(&opdata26)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs27, (TfLiteIntArray*)&outputs27, (TfLiteIntArray*)&inputs27, nullptr, const_cast<void*>(static_cast<const void*>(&opdata27)), nullptr, 142, },
  { (TfLiteIntArray*)&inputs28, (TfLiteIntArray*)&outputs28, (TfLiteIntArray*)&inputs28, nullptr, const_cast<void*>(static_cast<const void*>(&opdata28)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs29, (TfLiteIntArray*)&outputs29, (TfLiteIntArray*)&inputs29, nullptr, const_cast<void*>(static_cast<const void*>(&opdata29)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs30, (TfLiteIntArray*)&outputs30, (TfLiteIntArray*)&inputs30, nullptr, const_cast<void*>(static_cast<const void*>(&opdata30)), nullptr, 138, },
  { (TfLiteIntArray*)&inputs31, (TfLiteIntArray*)&outputs31, (TfLiteIntArray*)&inputs31, nullptr, const_cast<void*>(static_cast<const void*>(&opdata31)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs32, (TfLiteIntArray*)&outputs32, (TfLiteIntArray*)&inputs32, nullptr, const_cast<void*>(static_cast<const void*>(&opdata32)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs33, (TfLiteIntArray*)&outputs33, (TfLiteIntArray*)&inputs33, nullptr, const_cast<void*>(static_cast<const void*>(&opdata33)), nullptr, 142, },
  { (TfLiteIntArray*)&inputs34, (TfLiteIntArray*)&outputs34, (TfLiteIntArray*)&inputs34, nullptr, const_cast<void*>(static_cast<const void*>(&opdata34)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs35, (TfLiteIntArray*)&outputs35, (TfLiteIntArray*)&inputs35, nullptr, const_cast<void*>(static_cast<const void*>(&opdata35)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs36, (TfLiteIntArray*)&outputs36, (TfLiteIntArray*)&inputs36, nullptr, const_cast<void*>(static_cast<const void*>(&opdata36)), nullptr, 138, },
  { (TfLiteIntArray*)&inputs37, (TfLiteIntArray*)&outputs37, (TfLiteIntArray*)&inputs37, nullptr, const_cast<void*>(static_cast<const void*>(&opdata37)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs38, (TfLiteIntArray*)&outputs38, (TfLiteIntArray*)&inputs38, nullptr, const_cast<void*>(static_cast<const void*>(&opdata38)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs39, (TfLiteIntArray*)&outputs39, (TfLiteIntArray*)&inputs39, nullptr, const_cast<void*>(static_cast<const void*>(&opdata39)), nullptr, 142, },
  { (TfLiteIntArray*)&inputs40, (TfLiteIntArray*)&outputs40, (TfLiteIntArray*)&inputs40, nullptr, const_cast<void*>(static_cast<const void*>(&opdata40)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs41, (TfLiteIntArray*)&outputs41, (TfLiteIntArray*)&inputs41, nullptr, const_cast<void*>(static_cast<const void*>(&opdata41)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs42, (TfLiteIntArray*)&outputs42, (TfLiteIntArray*)&inputs42, nullptr, const_cast<void*>(static_cast<const void*>(&opdata42)), nullptr, 138, },
  { (TfLiteIntArray*)&inputs43, (TfLiteIntArray*)&outputs43, (TfLiteIntArray*)&inputs43, nullptr, const_cast<void*>(static_cast<const void*>(&opdata43)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs44, (TfLiteIntArray*)&outputs44, (TfLiteIntArray*)&inputs44, nullptr, const_cast<void*>(static_cast<const void*>(&opdata44)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs45, (TfLiteIntArray*)&outputs45, (TfLiteIntArray*)&inputs45, nullptr, const_cast<void*>(static_cast<const void*>(&opdata45)), nullptr, 142, },
  { (TfLiteIntArray*)&inputs46, (TfLiteIntArray*)&outputs46, (TfLiteIntArray*)&inputs46, nullptr, const_cast<void*>(static_cast<const void*>(&opdata46)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs47, (TfLiteIntArray*)&outputs47, (TfLiteIntArray*)&inputs47, nullptr, const_cast<void*>(static_cast<const void*>(&opdata47)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs48, (TfLiteIntArray*)&outputs48, (TfLiteIntArray*)&inputs48, nullptr, const_cast<void*>(static_cast<const void*>(&opdata48)), nullptr, 138, },
  { (TfLiteIntArray*)&inputs49, (TfLiteIntArray*)&outputs49, (TfLiteIntArray*)&inputs49, nullptr, const_cast<void*>(static_cast<const void*>(&opdata49)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs50, (TfLiteIntArray*)&outputs50, (TfLiteIntArray*)&inputs50, nullptr, const_cast<void*>(static_cast<const void*>(&opdata50)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs51, (TfLiteIntArray*)&outputs51, (TfLiteIntArray*)&inputs51, nullptr, const_cast<void*>(static_cast<const void*>(&opdata51)), nullptr, 142, },
  { (TfLiteIntArray*)&inputs52, (TfLiteIntArray*)&outputs52, (TfLiteIntArray*)&inputs52, nullptr, const_cast<void*>(static_cast<const void*>(&opdata52)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs53, (TfLiteIntArray*)&outputs53, (TfLiteIntArray*)&inputs53, nullptr, const_cast<void*>(static_cast<const void*>(&opdata53)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs54, (TfLiteIntArray*)&outputs54, (TfLiteIntArray*)&inputs54, nullptr, const_cast<void*>(static_cast<const void*>(&opdata54)), nullptr, 138, },
  { (TfLiteIntArray*)&inputs55, (TfLiteIntArray*)&outputs55, (TfLiteIntArray*)&inputs55, nullptr, const_cast<void*>(static_cast<const void*>(&opdata55)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs56, (TfLiteIntArray*)&outputs56, (TfLiteIntArray*)&inputs56, nullptr, const_cast<void*>(static_cast<const void*>(&opdata56)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs57, (TfLiteIntArray*)&outputs57, (TfLiteIntArray*)&inputs57, nullptr, const_cast<void*>(static_cast<const void*>(&opdata57)), nullptr, 142, },
  { (TfLiteIntArray*)&inputs58, (TfLiteIntArray*)&outputs58, (TfLiteIntArray*)&inputs58, nullptr, const_cast<void*>(static_cast<const void*>(&opdata58)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs59, (TfLiteIntArray*)&outputs59, (TfLiteIntArray*)&inputs59, nullptr, const_cast<void*>(static_cast<const void*>(&opdata59)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs60, (TfLiteIntArray*)&outputs60, (TfLiteIntArray*)&inputs60, nullptr, const_cast<void*>(static_cast<const void*>(&opdata60)), nullptr, 138, },
  { (TfLiteIntArray*)&inputs61, (TfLiteIntArray*)&outputs61, (TfLiteIntArray*)&inputs61, nullptr, const_cast<void*>(static_cast<const void*>(&opdata61)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs62, (TfLiteIntArray*)&outputs62, (TfLiteIntArray*)&inputs62, nullptr, const_cast<void*>(static_cast<const void*>(&opdata62)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs63, (TfLiteIntArray*)&outputs63, (TfLiteIntArray*)&inputs63, nullptr, const_cast<void*>(static_cast<const void*>(&opdata63)), nullptr, 142, },
  { (TfLiteIntArray*)&inputs64, (TfLiteIntArray*)&outputs64, (TfLiteIntArray*)&inputs64, nullptr, const_cast<void*>(static_cast<const void*>(&opdata64)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs65, (TfLiteIntArray*)&outputs65, (TfLiteIntArray*)&inputs65, nullptr, const_cast<void*>(static_cast<const void*>(&opdata65)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs66, (TfLiteIntArray*)&outputs66, (TfLiteIntArray*)&inputs66, nullptr, const_cast<void*>(static_cast<const void*>(&opdata66)), nullptr, 138, },
  { (TfLiteIntArray*)&inputs67, (TfLiteIntArray*)&outputs67, (TfLiteIntArray*)&inputs67, nullptr, const_cast<void*>(static_cast<const void*>(&opdata67)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs68, (TfLiteIntArray*)&outputs68, (TfLiteIntArray*)&inputs68, nullptr, const_cast<void*>(static_cast<const void*>(&opdata68)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs69, (TfLiteIntArray*)&outputs69, (TfLiteIntArray*)&inputs69, nullptr, const_cast<void*>(static_cast<const void*>(&opdata69)), nullptr, 142, },
  { (TfLiteIntArray*)&inputs70, (TfLiteIntArray*)&outputs70, (TfLiteIntArray*)&inputs70, nullptr, const_cast<void*>(static_cast<const void*>(&opdata70)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs71, (TfLiteIntArray*)&outputs71, (TfLiteIntArray*)&inputs71, nullptr, const_cast<void*>(static_cast<const void*>(&opdata71)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs72, (TfLiteIntArray*)&outputs72, (TfLiteIntArray*)&inputs72, nullptr, const_cast<void*>(static_cast<const void*>(&opdata72)), nullptr, 138, },
  { (TfLiteIntArray*)&inputs73, (TfLiteIntArray*)&outputs73, (TfLiteIntArray*)&inputs73, nullptr, const_cast<void*>(static_cast<const void*>(&opdata73)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs74, (TfLiteIntArray*)&outputs74, (TfLiteIntArray*)&inputs74, nullptr, const_cast<void*>(static_cast<const void*>(&opdata74)), nullptr, 57, },
  { (TfLiteIntArray*)&inputs75, (TfLiteIntArray*)&outputs75, (TfLiteIntArray*)&inputs75, nullptr, const_cast<void*>(static_cast<const void*>(&opdata75)), nullptr, 142, },
  { (TfLiteIntArray*)&inputs76, (TfLiteIntArray*)&outputs76, (TfLiteIntArray*)&inputs76, nullptr, const_cast<void*>(static_cast<const void*>(&opdata76)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs77, (TfLiteIntArray*)&outputs77, (TfLiteIntArray*)&inputs77, nullptr, const_cast<void*>(static_cast<const void*>(&opdata77)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs78, (TfLiteIntArray*)&outputs78, (TfLiteIntArray*)&inputs78, nullptr, const_cast<void*>(static_cast<const void*>(&opdata78)), nullptr, 138, },
  { (TfLiteIntArray*)&inputs79, (TfLiteIntArray*)&outputs79, (TfLiteIntArray*)&inputs79, nullptr, const_cast<void*>(static_cast<const void*>(&opdata79)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs80, (TfLiteIntArray*)&outputs80, (TfLiteIntArray*)&inputs80, nullptr, const_cast<void*>(static_cast<const void*>(&opdata80)), nullptr, 57, },
  { (TfLiteIntArray*)&inputs81, (TfLiteIntArray*)&outputs81, (TfLiteIntArray*)&inputs81, nullptr, const_cast<void*>(static_cast<const void*>(&opdata81)), nullptr, 142, },
  { (TfLiteIntArray*)&inputs82, (TfLiteIntArray*)&outputs82, (TfLiteIntArray*)&inputs82, nullptr, const_cast<void*>(static_cast<const void*>(&opdata82)), nullptr, 0, },
  { (TfLiteIntArray*)&inputs83, (TfLiteIntArray*)&outputs83, (TfLiteIntArray*)&inputs83, nullptr, const_cast<void*>(static_cast<const void*>(&opdata83)), nullptr, 61, },
  { (TfLiteIntArray*)&inputs84, (TfLiteIntArray*)&outputs84, (TfLiteIntArray*)&inputs84, nullptr, const_cast<void*>(static_cast<const void*>(&opdata84)), nullptr, 39, },
  { (TfLiteIntArray*)&inputs85, (TfLiteIntArray*)&outputs85, (TfLiteIntArray*)&inputs85, nullptr, const_cast<void*>(static_cast<const void*>(&opdata85)), nullptr, 168, },
  { (TfLiteIntArray*)&inputs86, (TfLiteIntArray*)&outputs86, (TfLiteIntArray*)&inputs86, nullptr, const_cast<void*>(static_cast<const void*>(&opdata86)), nullptr, 0, },
};
used_operators_e used_ops[] = {
OP_XC_pad_3_to_4, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_MEAN, OP_XC_pad, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_SOFTMAX, };


// Scratch buffer variables
int scratch_buffer_idx = 0;
const int scratch_buffer_offsets[29] = { 303632, 0, 100864, 0, 51072, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 261280 };
tflite::MicroContext mc;

// Xcore context and thread variables
xc_context_config_t xc_config;
constexpr int kStackWordsPerThread = 256;
constexpr int threadsStackSizeInUint64 = 1 * kStackWordsPerThread/2;
// We use uint64_t for xcThreadsStack so that it is aligned to 8 bytes
uint64_t xcThreadsStack[threadsStackSizeInUint64];

// Functions to be used as function pointers for TfLiteContext and MicroContext 
static void* AllocatePersistentBuffer(struct TfLiteContext* ctx,
                                                 size_t bytes) {
  static uint8_t *AllocPtr = tensor_arena + sizeof(tensor_arena);

  AllocPtr -= bytes;
  return AllocPtr;
}

static TfLiteEvalTensor *GetEvalTensor(const struct TfLiteContext *context,
                                       int tensor_idx) {
  return (TfLiteEvalTensor*)&tflTensors[tensor_idx];
}

static TfLiteStatus RequestScratchBufferInArena(struct TfLiteContext *context, size_t bytes,
                                       int *buffer_idx) {
  *buffer_idx = scratch_buffer_idx++;
  return kTfLiteOk;
};

static void *GetScratchBuffer(struct TfLiteContext *context,
                                       int buffer_idx) {
  return tensor_arena + scratch_buffer_offsets[buffer_idx];
}

static TfLiteTensor* AllocateTempInputTensor(const TfLiteNode* node, int index) {
      return &ctx.tensors[node->inputs->data[index]];
}

static TfLiteTensor* AllocateTempOutputTensor(const TfLiteNode* node, int index) {
      return &ctx.tensors[node->outputs->data[index]];
}

static void DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
}

static void* external_context() {
  return &xc_config;
}

} // namespace

TfLiteStatus model_init(void *flash_data) {
  // Set flash data in xcore context config
  xc_config.flash_data = flash_data;

  // Setup microcontext functions
  mc.AllocateTempInputTensor = &AllocateTempInputTensor;
  mc.AllocateTempOutputTensor = &AllocateTempOutputTensor;
  mc.DeallocateTempTfLiteTensor = &DeallocateTempTfLiteTensor;
  mc.external_context = &external_context;

  // Setup tflitecontext functions
  ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  ctx.GetEvalTensor = &GetEvalTensor;
  ctx.RequestScratchBufferInArena = &RequestScratchBufferInArena;
  ctx.GetScratchBuffer = &GetScratchBuffer;
  
  // Set microcontext as the context ptr
  ctx.impl_ = (void*)&mc;
  ctx.tensors = tflTensors;
  ctx.tensors_size = 117;
  registrations[OP_XC_pad_3_to_4] = *(tflite::ops::micro::xcore::Register_XC_pad_3_to_4());
  registrations[OP_XC_pad] = *(tflite::ops::micro::xcore::Register_XC_pad());
  registrations[OP_XC_ld_flash] = *(tflite::ops::micro::xcore::Register_XC_ld_flash());
  registrations[OP_XC_conv2d_v2] = *(tflite::ops::micro::xcore::Register_XC_conv2d_v2());
  registrations[OP_MEAN] = tflite::Register_MEAN();
  registrations[OP_SOFTMAX] = tflite::Register_SOFTMAX();


#ifdef TFLMC_XCORE_PROFILE
  printf("\nProfiling init()...");
  memset(op_times, 0, sizeof(op_times));
#endif

  for(size_t i = 0; i < 87; ++i) {
    if (registrations[used_ops[i]].init) {

#ifdef TFLMC_XCORE_PROFILE
      asm volatile ("gettime %0" : "=r" (time_t0));
#endif

      tflNodes[i].user_data = registrations[used_ops[i]].init(&ctx, (const char*)tflNodes[i].builtin_data, tflNodes[i].custom_initial_data_size);

#ifdef TFLMC_XCORE_PROFILE
      asm volatile ("gettime %0" : "=r" (time_t1));
      op_times[used_ops[i]] += time_t1 - time_t0;
      printf("\nnode %-5d %-32s %-12d", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

    }
  }

#ifdef TFLMC_XCORE_PROFILE
    printf("\n\nCumulative times for init()...");
    for(int i=0; i<OP_LAST; i++){
      printf("\n%-32s %-12d", op_strs[i], op_times[i]);
    }
  printf("\n");
  printf("\nProfiling prepare()...");
  memset(op_times, 0, sizeof(op_times));
#endif

  for(size_t i = 0; i < 87; ++i) {
    if (registrations[used_ops[i]].prepare) {

#ifdef TFLMC_XCORE_PROFILE
      asm volatile ("gettime %0" : "=r" (time_t0));
#endif

      TfLiteStatus status = registrations[used_ops[i]].prepare(&ctx, &tflNodes[i]);

#ifdef TFLMC_XCORE_PROFILE
      asm volatile ("gettime %0" : "=r" (time_t1));
      op_times[used_ops[i]] += time_t1 - time_t0;
      printf("\nnode %-5d %-32s %-12d", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

      if (status != kTfLiteOk) {
        return status;
      }
    }
  }

#ifdef TFLMC_XCORE_PROFILE
    printf("\n\nCumulative times for prepare()...");
    for(int i=0; i<OP_LAST; i++){
      printf("\n%-32s %-12d", op_strs[i], op_times[i]);
    }
  printf("\n");
#endif

  return kTfLiteOk;
}

static const int inTensorIndices[] = {
  0, 
};
TfLiteTensor* model_input(int index) {
  return &ctx.tensors[inTensorIndices[index]];
}

static const int outTensorIndices[] = {
  116, 
};
TfLiteTensor* model_output(int index) {
  return &ctx.tensors[outTensorIndices[index]];
}

#ifdef TFLMC_PRINT_TENSORS
unsigned char checksum(char *data, unsigned int length)
{
  static char sum;
  static char * end;
  sum = 0;
  end = data + length;

  do
  {
      sum -= *data++;
  } while (data != end);
  return sum;
}
#endif

TfLiteStatus model_invoke() {
  xc_config.thread_info.nstackwords = kStackWordsPerThread;
  xc_config.thread_info.stacks = &xcThreadsStack[threadsStackSizeInUint64 - 1];
  thread_init_1(&xc_config.thread_info);

#ifdef TFLMC_XCORE_PROFILE
  printf("\nProfiling invoke()...");
  memset(op_times, 0, sizeof(op_times));
  memset(op_counts, 0, sizeof(op_counts));
  op_times_summed = 0;
#endif

  for(size_t i = 0; i < 87; ++i) {

#ifdef TFLMC_PRINT_TENSORS
    // print every input tensor
    printf("\nnode in %d", i);
    for (int j=0; j<tflNodes[i].inputs->size; j++){
      printf("\ntensor %d, input %d, %d bytes, checksum %d\n", tflNodes[i].inputs->data[j], j, tflTensors[tflNodes[i].inputs->data[j]].bytes, checksum(tflTensors[tflNodes[i].inputs->data[j]].data.raw, tflTensors[tflNodes[i].inputs->data[j]].bytes));
      for(int k=0; k<tflTensors[tflNodes[i].inputs->data[j]].bytes; k++){
        printf("%d,", (int8_t)tflTensors[tflNodes[i].inputs->data[j]].data.raw[k]);
      }
    }
    printf("\n");
#endif

#ifdef TFLMC_XCORE_PROFILE
  asm volatile ("gettime %0" : "=r" (time_t0));
#endif

    TfLiteStatus status = registrations[used_ops[i]].invoke(&ctx, &tflNodes[i]);

#ifdef TFLMC_XCORE_PROFILE
  asm volatile ("gettime %0" : "=r" (time_t1));
  op_times[used_ops[i]] += time_t1 - time_t0;
  op_counts[used_ops[i]] += 1;
  printf("\nnode %-5d %-32s %-12d", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

#ifdef TFLMC_PRINT_TENSORS
    // print every output tensor
    printf("\nnode %d", i);
    for (int j=0; j<tflNodes[i].outputs->size; j++){
      printf("\ntensor %d, output %d, %d bytes, checksum %d\n", tflNodes[i].outputs->data[j], j, tflTensors[tflNodes[i].outputs->data[j]].bytes, checksum(tflTensors[tflNodes[i].outputs->data[j]].data.raw, tflTensors[tflNodes[i].outputs->data[j]].bytes));
      for(int k=0; k<tflTensors[tflNodes[i].outputs->data[j]].bytes; k++){
        printf("%d,", (int8_t)tflTensors[tflNodes[i].outputs->data[j]].data.raw[k]);
      }
    }
    printf("\n");
#endif

    if (status != kTfLiteOk) {
      thread_destroy(&xc_config.thread_info);
      return status;
    }
  }
  thread_destroy(&xc_config.thread_info);

#ifdef TFLMC_XCORE_PROFILE
  struct convopdata{
    const char * name;
    size_t thread_count;
    int evalStartTime;
    int threadsStartTime;
    int threadsDoneTime;
  };
  int conv_times1 = 0, conv_times2 = 0;
  printf("\n\nConv()...");
  for(size_t i = 0; i < 87; ++i) {
    if(used_ops[i] == OP_XC_conv2d_v2) {
      auto *op_data = reinterpret_cast<convopdata *>(tflNodes[i].user_data);
      conv_times1 += op_data->threadsStartTime - op_data->evalStartTime;
      conv_times2 += op_data->threadsDoneTime - op_data->threadsStartTime;
      printf("\nnode %-5d %-25s %-25s %-6d %-6d %-12d", i, op_strs[used_ops[i]], op_data->name, op_data->thread_count, op_data->threadsStartTime - op_data->evalStartTime, op_data->threadsDoneTime - op_data->threadsStartTime);
    }
  }
  printf("\nSummed - %-10d %-10d", conv_times1, conv_times2);

  printf("\n\nCumulative times for invoke()...");
  for(int i=0; i<OP_LAST; i++){
    op_times_summed += op_times[i];
    printf("\n%-5d %-32s %-12d %dms", op_counts[i], op_strs[i], op_times[i], op_times[i]/100000);
  }
  printf("\n\nTotal time for invoke() - %-10lld %lldms\n\n", op_times_summed, op_times_summed/100000);
#endif

  return kTfLiteOk;
}
