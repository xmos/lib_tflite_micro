// This file is generated. Do not edit.
// Generated on: 22.01.2023 12:19:51


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
extern TfLiteRegistration *Register_XC_ld_flash(void);
extern TfLiteRegistration *Register_XC_conv2d_v2(void);
} // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite

namespace {

constexpr int kTensorArenaSize = 417208;
uint8_t tensor_arena[kTensorArenaSize] ALIGN(8);
template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};
enum used_operators_e {
  OP_PAD, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_MEAN, OP_SOFTMAX,  OP_LAST
};

#ifdef TFLMC_XCORE_PROFILE
const char *op_strs[] = {
"OP_PAD", "OP_XC_ld_flash", "OP_XC_conv2d_v2", "OP_MEAN", "OP_SOFTMAX", };
int op_times[OP_LAST];
int op_counts[OP_LAST];
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
const ALIGN(8) int32_t tensor_data2[4*2] = { 
  0, 0, 
  0, 0, 
  0, 0, 
  0, 1, 
};
const TfArray<2, int> tensor_dimension2 = { 2, { 4,2 } };
const ALIGN(8) int16_t tensor_data3[32] = { 
    7200, 23617, 9167, 0, 0, 2882, 3711, 4417, 6784, 4371, 
    6134, 9039, 2989, 0, 4768, 4356, -3010, -2068, -4840, -4096, 
    -4096, 122, -1584, -946, 2793, -124, -1850, -1137, -69, -4096, 
    -1296, -259, 
};
const TfArray<1, int> tensor_dimension3 = { 1, { 32 } };
const ALIGN(8) int16_t tensor_data4[24] = { 
    1005, 1235, 720, 2642, 1342, 69, 642, 27586, -82, -117, 
    -1657, 4632, -66, 220, 30, -31838, 0, 0, 0, 0, 
    0, 0, 0, 0, 
};
const TfArray<1, int> tensor_dimension4 = { 1, { 24 } };
const ALIGN(8) int16_t tensor_data5[24] = { 
    106, 379, 2, 0, 402, 25577, 138, 0, -75, -81, 
    -235, -256, 82, -340, 7, -256, 0, 0, 0, 0, 
    0, 0, 0, 0, 
};
const TfArray<1, int> tensor_dimension5 = { 1, { 24 } };
const TfArray<4, int> tensor_dimension6 = { 4, { 1,224,224,4 } };
const TfArray<1, float> quant6_scale = { 1, { 0.0039215688593685627, } };
const TfArray<1, int> quant6_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant6 = { (TfLiteFloatArray*)&quant6_scale, (TfLiteIntArray*)&quant6_zero, 0 };
const TfArray<1, int> tensor_dimension7 = { 1, { 768 } };
const TfArray<4, int> tensor_dimension8 = { 4, { 1,112,112,8 } };
const TfArray<1, float> quant8_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant8_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant8 = { (TfLiteFloatArray*)&quant8_scale, (TfLiteIntArray*)&quant8_zero, 0 };
const TfArray<1, int> tensor_dimension9 = { 1, { 160 } };
const TfArray<4, int> tensor_dimension10 = { 4, { 1,112,112,8 } };
const TfArray<1, float> quant10_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant10_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant10 = { (TfLiteFloatArray*)&quant10_scale, (TfLiteIntArray*)&quant10_zero, 0 };
const TfArray<1, int> tensor_dimension11 = { 1, { 512 } };
const TfArray<4, int> tensor_dimension12 = { 4, { 1,112,112,16 } };
const TfArray<1, float> quant12_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant12_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant12 = { (TfLiteFloatArray*)&quant12_scale, (TfLiteIntArray*)&quant12_zero, 0 };
const TfArray<1, int> tensor_dimension13 = { 1, { 160 } };
const TfArray<1, int> tensor_dimension14 = { 1, { 56 } };
const TfArray<4, int> tensor_dimension15 = { 4, { 1,56,56,16 } };
const TfArray<1, float> quant15_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant15_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant15 = { (TfLiteFloatArray*)&quant15_scale, (TfLiteIntArray*)&quant15_zero, 0 };
const TfArray<1, int> tensor_dimension16 = { 1, { 768 } };
const TfArray<1, int> tensor_dimension17 = { 1, { 64 } };
const TfArray<4, int> tensor_dimension18 = { 4, { 1,56,56,32 } };
const TfArray<1, float> quant18_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant18_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant18 = { (TfLiteFloatArray*)&quant18_scale, (TfLiteIntArray*)&quant18_zero, 0 };
const TfArray<1, int> tensor_dimension19 = { 1, { 304 } };
const TfArray<1, int> tensor_dimension20 = { 1, { 96 } };
const TfArray<4, int> tensor_dimension21 = { 4, { 1,56,56,32 } };
const TfArray<1, float> quant21_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant21_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant21 = { (TfLiteFloatArray*)&quant21_scale, (TfLiteIntArray*)&quant21_zero, 0 };
const TfArray<1, int> tensor_dimension22 = { 1, { 1024 } };
const TfArray<1, int> tensor_dimension23 = { 1, { 64 } };
const TfArray<4, int> tensor_dimension24 = { 4, { 1,56,56,32 } };
const TfArray<1, float> quant24_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant24_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant24 = { (TfLiteFloatArray*)&quant24_scale, (TfLiteIntArray*)&quant24_zero, 0 };
const TfArray<1, int> tensor_dimension25 = { 1, { 304 } };
const TfArray<1, int> tensor_dimension26 = { 1, { 64 } };
const TfArray<4, int> tensor_dimension27 = { 4, { 1,28,28,32 } };
const TfArray<1, float> quant27_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant27_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant27 = { (TfLiteFloatArray*)&quant27_scale, (TfLiteIntArray*)&quant27_zero, 0 };
const TfArray<1, int> tensor_dimension28 = { 1, { 2048 } };
const TfArray<1, int> tensor_dimension29 = { 1, { 128 } };
const TfArray<4, int> tensor_dimension30 = { 4, { 1,28,28,64 } };
const TfArray<1, float> quant30_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant30_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant30 = { (TfLiteFloatArray*)&quant30_scale, (TfLiteIntArray*)&quant30_zero, 0 };
const TfArray<1, int> tensor_dimension31 = { 1, { 592 } };
const TfArray<1, int> tensor_dimension32 = { 1, { 128 } };
const TfArray<4, int> tensor_dimension33 = { 4, { 1,28,28,64 } };
const TfArray<1, float> quant33_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant33_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant33 = { (TfLiteFloatArray*)&quant33_scale, (TfLiteIntArray*)&quant33_zero, 0 };
const TfArray<1, int> tensor_dimension34 = { 1, { 4096 } };
const TfArray<1, int> tensor_dimension35 = { 1, { 128 } };
const TfArray<4, int> tensor_dimension36 = { 4, { 1,28,28,64 } };
const TfArray<1, float> quant36_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant36_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant36 = { (TfLiteFloatArray*)&quant36_scale, (TfLiteIntArray*)&quant36_zero, 0 };
const TfArray<1, int> tensor_dimension37 = { 1, { 592 } };
const TfArray<1, int> tensor_dimension38 = { 1, { 128 } };
const TfArray<4, int> tensor_dimension39 = { 4, { 1,14,14,64 } };
const TfArray<1, float> quant39_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant39_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant39 = { (TfLiteFloatArray*)&quant39_scale, (TfLiteIntArray*)&quant39_zero, 0 };
const TfArray<1, int> tensor_dimension40 = { 1, { 8192 } };
const TfArray<1, int> tensor_dimension41 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension42 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant42_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant42_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant42 = { (TfLiteFloatArray*)&quant42_scale, (TfLiteIntArray*)&quant42_zero, 0 };
const TfArray<1, int> tensor_dimension43 = { 1, { 1168 } };
const TfArray<1, int> tensor_dimension44 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension45 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant45_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant45_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant45 = { (TfLiteFloatArray*)&quant45_scale, (TfLiteIntArray*)&quant45_zero, 0 };
const TfArray<1, int> tensor_dimension46 = { 1, { 16384 } };
const TfArray<1, int> tensor_dimension47 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension48 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant48_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant48_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant48 = { (TfLiteFloatArray*)&quant48_scale, (TfLiteIntArray*)&quant48_zero, 0 };
const TfArray<1, int> tensor_dimension49 = { 1, { 1168 } };
const TfArray<1, int> tensor_dimension50 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension51 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant51_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant51_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant51 = { (TfLiteFloatArray*)&quant51_scale, (TfLiteIntArray*)&quant51_zero, 0 };
const TfArray<1, int> tensor_dimension52 = { 1, { 16384 } };
const TfArray<1, int> tensor_dimension53 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension54 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant54_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant54_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant54 = { (TfLiteFloatArray*)&quant54_scale, (TfLiteIntArray*)&quant54_zero, 0 };
const TfArray<1, int> tensor_dimension55 = { 1, { 1168 } };
const TfArray<1, int> tensor_dimension56 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension57 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant57_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant57_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant57 = { (TfLiteFloatArray*)&quant57_scale, (TfLiteIntArray*)&quant57_zero, 0 };
const TfArray<1, int> tensor_dimension58 = { 1, { 16384 } };
const TfArray<1, int> tensor_dimension59 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension60 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant60_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant60_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant60 = { (TfLiteFloatArray*)&quant60_scale, (TfLiteIntArray*)&quant60_zero, 0 };
const TfArray<1, int> tensor_dimension61 = { 1, { 1168 } };
const TfArray<1, int> tensor_dimension62 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension63 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant63_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant63_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant63 = { (TfLiteFloatArray*)&quant63_scale, (TfLiteIntArray*)&quant63_zero, 0 };
const TfArray<1, int> tensor_dimension64 = { 1, { 16384 } };
const TfArray<1, int> tensor_dimension65 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension66 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant66_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant66_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant66 = { (TfLiteFloatArray*)&quant66_scale, (TfLiteIntArray*)&quant66_zero, 0 };
const TfArray<1, int> tensor_dimension67 = { 1, { 1168 } };
const TfArray<1, int> tensor_dimension68 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension69 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant69_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant69_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant69 = { (TfLiteFloatArray*)&quant69_scale, (TfLiteIntArray*)&quant69_zero, 0 };
const TfArray<1, int> tensor_dimension70 = { 1, { 16384 } };
const TfArray<1, int> tensor_dimension71 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension72 = { 4, { 1,14,14,128 } };
const TfArray<1, float> quant72_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant72_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant72 = { (TfLiteFloatArray*)&quant72_scale, (TfLiteIntArray*)&quant72_zero, 0 };
const TfArray<1, int> tensor_dimension73 = { 1, { 1168 } };
const TfArray<1, int> tensor_dimension74 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension75 = { 4, { 1,7,7,128 } };
const TfArray<1, float> quant75_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant75_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant75 = { (TfLiteFloatArray*)&quant75_scale, (TfLiteIntArray*)&quant75_zero, 0 };
const TfArray<1, int> tensor_dimension76 = { 1, { 32768 } };
const TfArray<1, int> tensor_dimension77 = { 1, { 512 } };
const TfArray<4, int> tensor_dimension78 = { 4, { 1,7,7,256 } };
const TfArray<1, float> quant78_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant78_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant78 = { (TfLiteFloatArray*)&quant78_scale, (TfLiteIntArray*)&quant78_zero, 0 };
const TfArray<1, int> tensor_dimension79 = { 1, { 2320 } };
const TfArray<1, int> tensor_dimension80 = { 1, { 768 } };
const TfArray<4, int> tensor_dimension81 = { 4, { 1,7,7,256 } };
const TfArray<1, float> quant81_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant81_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant81 = { (TfLiteFloatArray*)&quant81_scale, (TfLiteIntArray*)&quant81_zero, 0 };
const TfArray<1, int> tensor_dimension82 = { 1, { 65536 } };
const TfArray<1, int> tensor_dimension83 = { 1, { 512 } };
const TfArray<4, int> tensor_dimension84 = { 4, { 1,7,7,256 } };
const TfArray<1, float> quant84_scale = { 1, { 0.023529412224888802, } };
const TfArray<1, int> quant84_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant84 = { (TfLiteFloatArray*)&quant84_scale, (TfLiteIntArray*)&quant84_zero, 0 };
const TfArray<4, int> tensor_dimension85 = { 4, { 1,1,1,256 } };
const TfArray<1, float> quant85_scale = { 1, { 0.022377394139766693, } };
const TfArray<1, int> quant85_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant85 = { (TfLiteFloatArray*)&quant85_scale, (TfLiteIntArray*)&quant85_zero, 0 };
const TfArray<1, int> tensor_dimension86 = { 1, { 256256 } };
const TfArray<1, int> tensor_dimension87 = { 1, { 2008 } };
const TfArray<2, int> tensor_dimension88 = { 2, { 1,1000 } };
const TfArray<1, float> quant88_scale = { 1, { 0.12677012383937836, } };
const TfArray<1, int> quant88_zero = { 1, { -31 } };
const TfLiteAffineQuantization quant88 = { (TfLiteFloatArray*)&quant88_scale, (TfLiteIntArray*)&quant88_zero, 0 };
const TfArray<2, int> tensor_dimension89 = { 2, { 1,1000 } };
const TfArray<1, float> quant89_scale = { 1, { 0.00390625, } };
const TfArray<1, int> quant89_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant89 = { (TfLiteFloatArray*)&quant89_scale, (TfLiteIntArray*)&quant89_zero, 0 };
const TfArray<2, int> inputs0 = { 2, { 0,2 } };
const TfArray<1, int> outputs0 = { 1, { 6 } };
uint8_t ALIGN(4) opdata1[45] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 1, 0, 0, 3, 5, 2, 18, 14, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 16, 83, 7, 0, 22, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs1 = 0; /* empty TfLiteIntArray */
const TfArray<1, int> outputs1 = { 1, { 7 } };
uint8_t ALIGN(4) opdata2[344] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 224, 0, 0, 0, 3, 0, 0, 0, 224, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 116, 3, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 128, 255, 255, 255, 128, 3, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 97, 103, 103, 112, 0, 8, 8, 0, 0, 0, 36, 0, 0, 0, 0, 111, 116, 112, 0, 8, 8, 0, 0, 0, 4, 0, 249, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 200, 2, 0, 0, 8, 0, 0, 0, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 23, 0, 0, 0, 46, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 200, 2, 0, 0, 8, 0, 0, 0, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 46, 0, 0, 0, 68, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 208, 2, 0, 0, 8, 0, 0, 0, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 68, 0, 0, 0, 90, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 208, 2, 0, 0, 8, 0, 0, 0, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 90, 0, 0, 0, 112, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 208, 2, 0, 0, 8, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 228, 0, 189, 0, 48, 1, 47, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 242, 0, 34, 0, 2, 0, 63, 1, 236, 0, 0, 0, 96, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs2 = { 3, { 6,7,5 } };
const TfArray<1, int> outputs2 = { 1, { 8 } };
uint8_t ALIGN(4) opdata3[45] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 1, 0, 160, 0, 5, 2, 18, 14, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 82, 7, 0, 22, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs3 = 0; /* empty TfLiteIntArray */
const TfArray<1, int> outputs3 = { 1, { 9 } };
uint8_t ALIGN(4) opdata4[356] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 112, 0, 0, 0, 3, 0, 0, 0, 112, 0, 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 104, 3, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 255, 255, 255, 128, 3, 0, 0, 8, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 111, 116, 112, 0, 8, 8, 0, 0, 0, 2, 0, 249, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 200, 2, 0, 0, 8, 0, 0, 0, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 23, 0, 0, 0, 46, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 200, 2, 0, 0, 8, 0, 0, 0, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 46, 0, 0, 0, 68, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 208, 2, 0, 0, 8, 0, 0, 0, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 68, 0, 0, 0, 90, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 208, 2, 0, 0, 8, 0, 0, 0, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 90, 0, 0, 0, 112, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 208, 2, 0, 0, 8, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 240, 0, 189, 0, 60, 1, 59, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 254, 0, 34, 0, 4, 0, 75, 1, 236, 0, 0, 0, 176, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs4 = { 3, { 8,9,4 } };
const TfArray<1, int> outputs4 = { 1, { 10 } };
uint8_t ALIGN(4) opdata5[45] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 1, 0, 0, 2, 5, 2, 18, 14, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 80, 7, 0, 22, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs5 = 0; /* empty TfLiteIntArray */
const TfArray<1, int> outputs5 = { 1, { 11 } };
uint8_t ALIGN(4) opdata6[312] = { 107, 116, 0, 109, 112, 0, 32, 128, 3, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 232, 255, 255, 255, 120, 3, 0, 0, 0, 97, 103, 103, 112, 0, 8, 16, 0, 0, 0, 8, 0, 0, 0, 0, 111, 116, 112, 0, 8, 16, 0, 0, 0, 0, 0, 253, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 0, 0, 0, 0, 23, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 144, 5, 0, 0, 16, 0, 0, 0, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 23, 0, 0, 0, 46, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 144, 5, 0, 0, 16, 0, 0, 0, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 46, 0, 0, 0, 68, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 16, 0, 0, 0, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 68, 0, 0, 0, 90, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 16, 0, 0, 0, 0, 32, 0, 0, 0, 0, 112, 0, 0, 0, 90, 0, 0, 0, 112, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 16, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 228, 0, 189, 0, 16, 1, 15, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 242, 0, 34, 0, 1, 0, 31, 1, 236, 0, 0, 0, 64, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs6 = { 3, { 10,11,3 } };
const TfArray<1, int> outputs6 = { 1, { 12 } };
uint8_t ALIGN(4) opdata7[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 160, 0, 112, 0, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 96, 79, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs7 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs7 = { 2, { 13,14 } };
uint8_t ALIGN(4) opdata8[356] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 112, 0, 0, 0, 3, 0, 0, 0, 112, 0, 0, 0, 16, 0, 0, 0, 1, 0, 0, 0, 208, 6, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 255, 255, 255, 0, 7, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 111, 116, 112, 0, 8, 16, 0, 0, 0, 248, 255, 0, 0, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 192, 2, 0, 0, 16, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 12, 0, 0, 0, 23, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 208, 2, 0, 0, 16, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 23, 0, 0, 0, 34, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 208, 2, 0, 0, 16, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 34, 0, 0, 0, 45, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 208, 2, 0, 0, 16, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 45, 0, 0, 0, 56, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 208, 2, 0, 0, 16, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 240, 0, 189, 0, 60, 1, 59, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 254, 0, 34, 0, 4, 0, 75, 1, 236, 0, 1, 0, 176, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs8 = { 3, { 12,13,14 } };
const TfArray<1, int> outputs8 = { 1, { 15 } };
uint8_t ALIGN(4) opdata9[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 3, 128, 0, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 224, 75, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs9 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs9 = { 2, { 16,17 } };
uint8_t ALIGN(4) opdata10[312] = { 107, 116, 0, 109, 112, 0, 32, 128, 3, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 240, 255, 255, 255, 112, 3, 0, 0, 0, 97, 103, 103, 112, 0, 8, 32, 0, 0, 0, 16, 0, 0, 0, 0, 111, 116, 112, 0, 8, 32, 0, 0, 0, 1, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 12, 0, 0, 0, 23, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 23, 0, 0, 0, 34, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 34, 0, 0, 0, 45, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 45, 0, 0, 0, 56, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 32, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 228, 0, 189, 0, 16, 1, 15, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 242, 0, 34, 0, 1, 0, 31, 1, 236, 0, 0, 0, 64, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs10 = { 3, { 15,16,17 } };
const TfArray<1, int> outputs10 = { 1, { 18 } };
uint8_t ALIGN(4) opdata11[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 48, 1, 192, 0, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 240, 73, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs11 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs11 = { 2, { 19,20 } };
uint8_t ALIGN(4) opdata12[356] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 56, 0, 0, 0, 3, 0, 0, 0, 56, 0, 0, 0, 32, 0, 0, 0, 1, 0, 0, 0, 160, 6, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 255, 255, 255, 0, 7, 0, 0, 32, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 111, 116, 112, 0, 8, 32, 0, 0, 0, 249, 255, 0, 0, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 12, 0, 0, 0, 23, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 23, 0, 0, 0, 34, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 34, 0, 0, 0, 45, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 45, 0, 0, 0, 56, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 32, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 240, 0, 189, 0, 60, 1, 59, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 254, 0, 34, 0, 4, 0, 75, 1, 236, 0, 1, 0, 176, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs12 = { 3, { 18,19,20 } };
const TfArray<1, int> outputs12 = { 1, { 21 } };
uint8_t ALIGN(4) opdata13[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 4, 128, 0, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 69, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs13 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs13 = { 2, { 22,23 } };
uint8_t ALIGN(4) opdata14[304] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 32, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 224, 6, 0, 0, 0, 111, 116, 112, 0, 8, 32, 0, 0, 0, 2, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 12, 0, 0, 0, 23, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 23, 0, 0, 0, 34, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 34, 0, 0, 0, 45, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 56, 0, 0, 0, 45, 0, 0, 0, 56, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 160, 5, 0, 0, 32, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 244, 0, 189, 0, 8, 1, 7, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 2, 1, 34, 0, 0, 0, 23, 1, 236, 0, 0, 0, 0, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs14 = { 3, { 21,22,23 } };
const TfArray<1, int> outputs14 = { 1, { 24 } };
uint8_t ALIGN(4) opdata15[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 48, 1, 128, 0, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 192, 67, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs15 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs15 = { 2, { 25,26 } };
uint8_t ALIGN(4) opdata16[356] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 56, 0, 0, 0, 3, 0, 0, 0, 56, 0, 0, 0, 32, 0, 0, 0, 1, 0, 0, 0, 160, 6, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 255, 255, 255, 0, 7, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 111, 116, 112, 0, 8, 32, 0, 0, 0, 2, 0, 251, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 192, 2, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 6, 0, 0, 0, 12, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 192, 2, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 12, 0, 0, 0, 18, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 192, 2, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 18, 0, 0, 0, 23, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 224, 2, 0, 0, 32, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 23, 0, 0, 0, 28, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 224, 2, 0, 0, 32, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 240, 0, 189, 0, 60, 1, 59, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 254, 0, 34, 0, 4, 0, 75, 1, 236, 0, 0, 0, 176, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs16 = { 3, { 24,25,26 } };
const TfArray<1, int> outputs16 = { 1, { 27 } };
uint8_t ALIGN(4) opdata17[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 8, 0, 1, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 192, 58, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs17 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs17 = { 2, { 28,29 } };
uint8_t ALIGN(4) opdata18[304] = { 107, 116, 0, 109, 112, 0, 8, 128, 3, 0, 0, 32, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 96, 3, 0, 0, 0, 111, 116, 112, 0, 8, 64, 0, 0, 0, 3, 0, 251, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 6, 0, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 12, 0, 0, 0, 18, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 18, 0, 0, 0, 23, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 192, 5, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 23, 0, 0, 0, 28, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 192, 5, 0, 0, 64, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 244, 0, 189, 0, 8, 1, 7, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 2, 1, 34, 0, 0, 0, 23, 1, 236, 0, 0, 0, 0, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs18 = { 3, { 27,28,29 } };
const TfArray<1, int> outputs18 = { 1, { 30 } };
uint8_t ALIGN(4) opdata19[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 80, 2, 0, 1, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 55, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs19 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs19 = { 2, { 31,32 } };
uint8_t ALIGN(4) opdata20[356] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 28, 0, 0, 0, 3, 0, 0, 0, 28, 0, 0, 0, 64, 0, 0, 0, 1, 0, 0, 0, 64, 6, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 255, 255, 255, 0, 7, 0, 0, 64, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 111, 116, 112, 0, 8, 64, 0, 0, 0, 1, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 6, 0, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 12, 0, 0, 0, 18, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 18, 0, 0, 0, 23, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 192, 5, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 23, 0, 0, 0, 28, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 192, 5, 0, 0, 64, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 240, 0, 189, 0, 60, 1, 59, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 254, 0, 34, 0, 4, 0, 75, 1, 236, 0, 0, 0, 176, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs20 = { 3, { 30,31,32 } };
const TfArray<1, int> outputs20 = { 1, { 33 } };
uint8_t ALIGN(4) opdata21[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 16, 0, 1, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 38, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs21 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs21 = { 2, { 34,35 } };
uint8_t ALIGN(4) opdata22[304] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 64, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 192, 6, 0, 0, 0, 111, 116, 112, 0, 8, 64, 0, 0, 0, 3, 0, 250, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 6, 0, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 12, 0, 0, 0, 18, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 18, 0, 0, 0, 23, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 192, 5, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 28, 0, 0, 0, 23, 0, 0, 0, 28, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 192, 5, 0, 0, 64, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 244, 0, 189, 0, 8, 1, 7, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 2, 1, 34, 0, 0, 0, 23, 1, 236, 0, 0, 0, 0, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs22 = { 3, { 33,34,35 } };
const TfArray<1, int> outputs22 = { 1, { 36 } };
uint8_t ALIGN(4) opdata23[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 80, 2, 0, 1, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 32, 35, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs23 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs23 = { 2, { 37,38 } };
uint8_t ALIGN(4) opdata24[356] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 28, 0, 0, 0, 3, 0, 0, 0, 28, 0, 0, 0, 64, 0, 0, 0, 1, 0, 0, 0, 64, 6, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 255, 255, 255, 0, 7, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 111, 116, 112, 0, 8, 64, 0, 0, 0, 2, 0, 250, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 192, 2, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 192, 2, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 192, 2, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 192, 2, 0, 0, 64, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 64, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 240, 0, 189, 0, 60, 1, 59, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 254, 0, 34, 0, 4, 0, 75, 1, 236, 0, 0, 0, 176, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs24 = { 3, { 36,37,38 } };
const TfArray<1, int> outputs24 = { 1, { 39 } };
uint8_t ALIGN(4) opdata25[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 32, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 32, 1, 7, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs25 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs25 = { 2, { 40,41 } };
uint8_t ALIGN(4) opdata26[304] = { 107, 116, 0, 109, 112, 0, 8, 128, 3, 0, 0, 64, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 64, 3, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 4, 0, 251, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 128, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 244, 0, 189, 0, 8, 1, 7, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 2, 1, 34, 0, 0, 0, 23, 1, 236, 0, 0, 0, 0, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs26 = { 3, { 39,40,41 } };
const TfArray<1, int> outputs26 = { 1, { 42 } };
uint8_t ALIGN(4) opdata27[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 144, 4, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 144, 250, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs27 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs27 = { 2, { 43,44 } };
uint8_t ALIGN(4) opdata28[356] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 128, 0, 0, 0, 1, 0, 0, 0, 128, 5, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 255, 255, 255, 0, 7, 0, 0, 128, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 1, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 128, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 240, 0, 189, 0, 60, 1, 59, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 254, 0, 34, 0, 4, 0, 75, 1, 236, 0, 0, 0, 176, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs28 = { 3, { 42,43,44 } };
const TfArray<1, int> outputs28 = { 1, { 45 } };
uint8_t ALIGN(4) opdata29[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 64, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 144, 184, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs29 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs29 = { 2, { 46,47 } };
uint8_t ALIGN(4) opdata30[304] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 3, 0, 253, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 128, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 244, 0, 189, 0, 8, 1, 7, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 2, 1, 34, 0, 0, 0, 23, 1, 236, 0, 0, 0, 0, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs30 = { 3, { 45,46,47 } };
const TfArray<1, int> outputs30 = { 1, { 48 } };
uint8_t ALIGN(4) opdata31[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 144, 4, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 178, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs31 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs31 = { 2, { 49,50 } };
uint8_t ALIGN(4) opdata32[356] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 128, 0, 0, 0, 1, 0, 0, 0, 128, 5, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 255, 255, 255, 0, 7, 0, 0, 128, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 2, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 128, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 240, 0, 189, 0, 60, 1, 59, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 254, 0, 34, 0, 4, 0, 75, 1, 236, 0, 0, 0, 176, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs32 = { 3, { 48,49,50 } };
const TfArray<1, int> outputs32 = { 1, { 51 } };
uint8_t ALIGN(4) opdata33[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 64, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 112, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs33 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs33 = { 2, { 52,53 } };
uint8_t ALIGN(4) opdata34[304] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 4, 0, 251, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 128, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 244, 0, 189, 0, 8, 1, 7, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 2, 1, 34, 0, 0, 0, 23, 1, 236, 0, 0, 0, 0, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs34 = { 3, { 51,52,53 } };
const TfArray<1, int> outputs34 = { 1, { 54 } };
uint8_t ALIGN(4) opdata35[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 144, 4, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 105, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs35 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs35 = { 2, { 55,56 } };
uint8_t ALIGN(4) opdata36[356] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 128, 0, 0, 0, 1, 0, 0, 0, 128, 5, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 255, 255, 255, 0, 7, 0, 0, 128, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 2, 0, 250, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 128, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 240, 0, 189, 0, 60, 1, 59, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 254, 0, 34, 0, 4, 0, 75, 1, 236, 0, 0, 0, 176, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs36 = { 3, { 54,55,56 } };
const TfArray<1, int> outputs36 = { 1, { 57 } };
uint8_t ALIGN(4) opdata37[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 64, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 112, 39, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs37 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs37 = { 2, { 58,59 } };
uint8_t ALIGN(4) opdata38[304] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 3, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 128, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 244, 0, 189, 0, 8, 1, 7, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 2, 1, 34, 0, 0, 0, 23, 1, 236, 0, 0, 0, 0, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs38 = { 3, { 57,58,59 } };
const TfArray<1, int> outputs38 = { 1, { 60 } };
uint8_t ALIGN(4) opdata39[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 144, 4, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 224, 32, 6, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs39 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs39 = { 2, { 61,62 } };
uint8_t ALIGN(4) opdata40[356] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 128, 0, 0, 0, 1, 0, 0, 0, 128, 5, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 255, 255, 255, 0, 7, 0, 0, 128, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 2, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 128, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 240, 0, 189, 0, 60, 1, 59, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 254, 0, 34, 0, 4, 0, 75, 1, 236, 0, 0, 0, 176, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs40 = { 3, { 60,61,62 } };
const TfArray<1, int> outputs40 = { 1, { 63 } };
uint8_t ALIGN(4) opdata41[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 64, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 224, 222, 5, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs41 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs41 = { 2, { 64,65 } };
uint8_t ALIGN(4) opdata42[304] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 3, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 128, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 244, 0, 189, 0, 8, 1, 7, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 2, 1, 34, 0, 0, 0, 23, 1, 236, 0, 0, 0, 0, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs42 = { 3, { 63,64,65 } };
const TfArray<1, int> outputs42 = { 1, { 66 } };
uint8_t ALIGN(4) opdata43[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 144, 4, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 80, 216, 5, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs43 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs43 = { 2, { 67,68 } };
uint8_t ALIGN(4) opdata44[356] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 128, 0, 0, 0, 1, 0, 0, 0, 128, 5, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 255, 255, 255, 0, 7, 0, 0, 128, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 2, 0, 253, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 128, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 240, 0, 189, 0, 60, 1, 59, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 254, 0, 34, 0, 4, 0, 75, 1, 236, 0, 0, 0, 176, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs44 = { 3, { 66,67,68 } };
const TfArray<1, int> outputs44 = { 1, { 69 } };
uint8_t ALIGN(4) opdata45[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 64, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 80, 150, 5, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs45 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs45 = { 2, { 70,71 } };
uint8_t ALIGN(4) opdata46[304] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 128, 6, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 4, 0, 250, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 6, 0, 0, 0, 9, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 9, 0, 0, 0, 12, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 5, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 14, 0, 0, 0, 12, 0, 0, 0, 14, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 128, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 244, 0, 189, 0, 8, 1, 7, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 2, 1, 34, 0, 0, 0, 23, 1, 236, 0, 0, 0, 0, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs46 = { 3, { 69,70,71 } };
const TfArray<1, int> outputs46 = { 1, { 72 } };
uint8_t ALIGN(4) opdata47[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 144, 4, 0, 2, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 192, 143, 5, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs47 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs47 = { 2, { 73,74 } };
uint8_t ALIGN(4) opdata48[356] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 14, 0, 0, 0, 3, 0, 0, 0, 14, 0, 0, 0, 128, 0, 0, 0, 1, 0, 0, 0, 128, 5, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 255, 255, 255, 0, 7, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 111, 116, 112, 0, 8, 128, 0, 0, 0, 2, 0, 252, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 2, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 128, 2, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 128, 0, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 128, 0, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 240, 0, 189, 0, 60, 1, 59, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 254, 0, 34, 0, 4, 0, 75, 1, 236, 0, 0, 0, 176, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs48 = { 3, { 72,73,74 } };
const TfArray<1, int> outputs48 = { 1, { 75 } };
uint8_t ALIGN(4) opdata49[57] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 0, 0, 128, 0, 0, 0, 4, 0, 0, 6, 6, 2, 27, 23, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 192, 11, 5, 0, 32, 0, 0, 0, 6, 42, 10, 38, 1,  }; /* custom_initial_data */
const int inputs49 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs49 = { 2, { 76,77 } };
uint8_t ALIGN(4) opdata50[304] = { 107, 116, 0, 109, 112, 0, 8, 128, 3, 0, 0, 128, 0, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 111, 116, 112, 0, 8, 0, 1, 0, 0, 3, 0, 250, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 244, 0, 189, 0, 8, 1, 7, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 2, 1, 34, 0, 0, 0, 23, 1, 236, 0, 0, 0, 0, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs50 = { 3, { 75,76,77 } };
const TfArray<1, int> outputs50 = { 1, { 78 } };
uint8_t ALIGN(4) opdata51[49] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 16, 9, 0, 6, 5, 5, 2, 21, 17, 0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 176, 252, 4, 0, 26, 0, 0, 0, 6, 41, 10, 38, 1,  }; /* custom_initial_data */
const int inputs51 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs51 = { 2, { 79,80 } };
uint8_t ALIGN(4) opdata52[356] = { 107, 116, 0, 109, 112, 0, 64, 3, 0, 0, 0, 7, 0, 0, 0, 3, 0, 0, 0, 7, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 128, 255, 255, 255, 0, 7, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 97, 103, 103, 112, 0, 20, 144, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 111, 116, 112, 0, 8, 0, 1, 0, 0, 249, 255, 0, 0, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 240, 0, 189, 0, 60, 1, 59, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 254, 0, 34, 0, 4, 0, 75, 1, 236, 0, 1, 0, 176, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs52 = { 3, { 78,79,80 } };
const TfArray<1, int> outputs52 = { 1, { 81 } };
uint8_t ALIGN(4) opdata53[57] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0, 6, 6, 2, 27, 23, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 176, 248, 3, 0, 32, 0, 0, 0, 6, 42, 10, 38, 1,  }; /* custom_initial_data */
const int inputs53 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs53 = { 2, { 82,83 } };
uint8_t ALIGN(4) opdata54[304] = { 107, 116, 0, 109, 112, 0, 8, 0, 7, 0, 0, 0, 1, 0, 0, 0, 97, 103, 103, 112, 0, 24, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 111, 116, 112, 0, 8, 0, 1, 0, 0, 3, 0, 249, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 5, 0, 0, 0, 6, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 32, 0, 0, 0, 0, 7, 0, 0, 0, 6, 0, 0, 0, 7, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 1, 0, 0, 0, 5, 170, 137, 104, 71, 38, 20, 20, 20, 20, 20, 7, 0, 244, 0, 189, 0, 8, 1, 7, 1, 221, 0, 209, 0, 207, 0, 14, 0, 2, 0, 7, 0, 2, 1, 34, 0, 0, 0, 23, 1, 236, 0, 0, 0, 0, 0, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs54 = { 3, { 81,82,83 } };
const TfArray<1, int> outputs54 = { 1, { 84 } };
const TfLiteReducerParams opdata55 = { true };
const TfArray<2, int> inputs55 = { 2, { 84,1 } };
const TfArray<1, int> outputs55 = { 1, { 85 } };
uint8_t ALIGN(4) opdata56[39] = { 97, 100, 100, 114, 0, 115, 105, 122, 101, 115, 0, 0, 2, 0, 0, 0, 0, 233, 3, 0, 176, 15, 0, 0, 6, 6, 2, 27, 23, 2, 1, 2, 0, 17, 4, 42, 4, 36, 1,  }; /* custom_initial_data */
const int inputs56 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs56 = { 2, { 86,87 } };
uint8_t ALIGN(4) opdata57[160] = { 107, 116, 0, 109, 112, 0, 32, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 97, 103, 103, 112, 0, 8, 232, 3, 0, 0, 0, 1, 0, 0, 0, 111, 116, 112, 0, 8, 232, 3, 0, 0, 4, 0, 254, 255, 0, 111, 116, 116, 0, 115, 99, 114, 97, 116, 99, 104, 0, 97, 107, 112, 0, 32, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 232, 3, 0, 0, 0, 1, 34, 20, 7, 83, 43, 125, 123, 72, 59, 56, 7, 0, 1, 0, 7, 0, 90, 0, 18, 0, 1, 0, 135, 0, 84, 0, 0, 0, 32, 1, 20, 40, 5, 20, 20, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<3, int> inputs57 = { 3, { 85,86,87 } };
const TfArray<1, int> outputs57 = { 1, { 88 } };
const TfLiteSoftmaxParams opdata58 = { 1 };
const TfArray<1, int> inputs58 = { 1, { 88 } };
const TfArray<1, int> outputs58 = { 1, { 89 } };
TfLiteTensor tflTensors[] = {
  { {(int32_t*)(tensor_arena + 200704)},(TfLiteIntArray*)&tensor_dimension0, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant0)) }, {quant0.scale->data[0], quant0.zero_point->data[0] },150528, kTfLiteArenaRw, false, },
  { {(int32_t*)tensor_data1},(TfLiteIntArray*)&tensor_dimension1, kTfLiteInt32, {kTfLiteNoQuantization, nullptr }, {0,0},8, kTfLiteMmapRo, false, },
  { {(int32_t*)tensor_data2},(TfLiteIntArray*)&tensor_dimension2, kTfLiteInt32, {kTfLiteNoQuantization, nullptr }, {0,0},32, kTfLiteMmapRo, false, },
  { {(int32_t*)tensor_data3},(TfLiteIntArray*)&tensor_dimension3, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},64, kTfLiteMmapRo, false, },
  { {(int32_t*)tensor_data4},(TfLiteIntArray*)&tensor_dimension4, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},48, kTfLiteMmapRo, false, },
  { {(int32_t*)tensor_data5},(TfLiteIntArray*)&tensor_dimension5, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},48, kTfLiteMmapRo, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension6, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant6)) }, {quant6.scale->data[0], quant6.zero_point->data[0] },200704, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 200704)},(TfLiteIntArray*)&tensor_dimension7, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},768, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 301056)},(TfLiteIntArray*)&tensor_dimension8, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant8)) }, {quant8.scale->data[0], quant8.zero_point->data[0] },100352, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 880)},(TfLiteIntArray*)&tensor_dimension9, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},160, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 200704)},(TfLiteIntArray*)&tensor_dimension10, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant10)) }, {quant10.scale->data[0], quant10.zero_point->data[0] },100352, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 301056)},(TfLiteIntArray*)&tensor_dimension11, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension12, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant12)) }, {quant12.scale->data[0], quant12.zero_point->data[0] },200704, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 251760)},(TfLiteIntArray*)&tensor_dimension13, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},160, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 251920)},(TfLiteIntArray*)&tensor_dimension14, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},112, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 200704)},(TfLiteIntArray*)&tensor_dimension15, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant15)) }, {quant15.scale->data[0], quant15.zero_point->data[0] },50176, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 100352)},(TfLiteIntArray*)&tensor_dimension16, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},768, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 101120)},(TfLiteIntArray*)&tensor_dimension17, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},128, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension18, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant18)) }, {quant18.scale->data[0], quant18.zero_point->data[0] },100352, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 200704)},(TfLiteIntArray*)&tensor_dimension19, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},304, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 201008)},(TfLiteIntArray*)&tensor_dimension20, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},192, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 100352)},(TfLiteIntArray*)&tensor_dimension21, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant21)) }, {quant21.scale->data[0], quant21.zero_point->data[0] },100352, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 200704)},(TfLiteIntArray*)&tensor_dimension22, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1024, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 201728)},(TfLiteIntArray*)&tensor_dimension23, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},128, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension24, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant24)) }, {quant24.scale->data[0], quant24.zero_point->data[0] },100352, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 125440)},(TfLiteIntArray*)&tensor_dimension25, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},304, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 126624)},(TfLiteIntArray*)&tensor_dimension26, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},128, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 100352)},(TfLiteIntArray*)&tensor_dimension27, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant27)) }, {quant27.scale->data[0], quant27.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension28, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},2048, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 52224)},(TfLiteIntArray*)&tensor_dimension29, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},256, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension30, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant30)) }, {quant30.scale->data[0], quant30.zero_point->data[0] },50176, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 100352)},(TfLiteIntArray*)&tensor_dimension31, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},592, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 100944)},(TfLiteIntArray*)&tensor_dimension32, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},256, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension33, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant33)) }, {quant33.scale->data[0], quant33.zero_point->data[0] },50176, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 100352)},(TfLiteIntArray*)&tensor_dimension34, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},4096, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 104448)},(TfLiteIntArray*)&tensor_dimension35, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},256, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension36, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant36)) }, {quant36.scale->data[0], quant36.zero_point->data[0] },50176, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 62720)},(TfLiteIntArray*)&tensor_dimension37, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},592, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 63312)},(TfLiteIntArray*)&tensor_dimension38, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},256, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension39, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant39)) }, {quant39.scale->data[0], quant39.zero_point->data[0] },12544, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 25088)},(TfLiteIntArray*)&tensor_dimension40, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},8192, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 33280)},(TfLiteIntArray*)&tensor_dimension41, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension42, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant42)) }, {quant42.scale->data[0], quant42.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension43, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1168, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 51344)},(TfLiteIntArray*)&tensor_dimension44, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 25088)},(TfLiteIntArray*)&tensor_dimension45, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant45)) }, {quant45.scale->data[0], quant45.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension46, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},16384, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 66560)},(TfLiteIntArray*)&tensor_dimension47, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension48, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant48)) }, {quant48.scale->data[0], quant48.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension49, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1168, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 51344)},(TfLiteIntArray*)&tensor_dimension50, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 25088)},(TfLiteIntArray*)&tensor_dimension51, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant51)) }, {quant51.scale->data[0], quant51.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension52, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},16384, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 66560)},(TfLiteIntArray*)&tensor_dimension53, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension54, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant54)) }, {quant54.scale->data[0], quant54.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension55, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1168, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 51344)},(TfLiteIntArray*)&tensor_dimension56, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 25088)},(TfLiteIntArray*)&tensor_dimension57, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant57)) }, {quant57.scale->data[0], quant57.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension58, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},16384, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 66560)},(TfLiteIntArray*)&tensor_dimension59, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension60, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant60)) }, {quant60.scale->data[0], quant60.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension61, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1168, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 51344)},(TfLiteIntArray*)&tensor_dimension62, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 25088)},(TfLiteIntArray*)&tensor_dimension63, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant63)) }, {quant63.scale->data[0], quant63.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension64, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},16384, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 66560)},(TfLiteIntArray*)&tensor_dimension65, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension66, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant66)) }, {quant66.scale->data[0], quant66.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension67, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1168, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 51344)},(TfLiteIntArray*)&tensor_dimension68, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 25088)},(TfLiteIntArray*)&tensor_dimension69, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant69)) }, {quant69.scale->data[0], quant69.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 50176)},(TfLiteIntArray*)&tensor_dimension70, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},16384, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 66560)},(TfLiteIntArray*)&tensor_dimension71, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension72, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant72)) }, {quant72.scale->data[0], quant72.zero_point->data[0] },25088, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 25088)},(TfLiteIntArray*)&tensor_dimension73, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},1168, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 26256)},(TfLiteIntArray*)&tensor_dimension74, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},512, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 45312)},(TfLiteIntArray*)&tensor_dimension75, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant75)) }, {quant75.scale->data[0], quant75.zero_point->data[0] },6272, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension76, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},32768, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 51584)},(TfLiteIntArray*)&tensor_dimension77, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},1024, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&tensor_dimension78, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant78)) }, {quant78.scale->data[0], quant78.zero_point->data[0] },12544, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension79, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},2320, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 2320)},(TfLiteIntArray*)&tensor_dimension80, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},1536, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 78080)},(TfLiteIntArray*)&tensor_dimension81, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant81)) }, {quant81.scale->data[0], quant81.zero_point->data[0] },12544, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension82, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},65536, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 90624)},(TfLiteIntArray*)&tensor_dimension83, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},1024, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 65536)},(TfLiteIntArray*)&tensor_dimension84, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant84)) }, {quant84.scale->data[0], quant84.zero_point->data[0] },12544, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 261568)},(TfLiteIntArray*)&tensor_dimension85, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant85)) }, {quant85.scale->data[0], quant85.zero_point->data[0] },256, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension86, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},256256, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 256256)},(TfLiteIntArray*)&tensor_dimension87, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},4016, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 260272)},(TfLiteIntArray*)&tensor_dimension88, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant88)) }, {quant88.scale->data[0], quant88.zero_point->data[0] },1000, kTfLiteArenaRw, false, },
  { {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&tensor_dimension89, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&quant89)) }, {quant89.scale->data[0], quant89.zero_point->data[0] },1000, kTfLiteArenaRw, false, },
};
TfLiteNode tflNodes[] = {
  { (TfLiteIntArray*)&inputs0, (TfLiteIntArray*)&outputs0, (TfLiteIntArray*)&inputs0, nullptr, nullptr, nullptr, 0, },
  { (TfLiteIntArray*)&inputs1, (TfLiteIntArray*)&outputs1, (TfLiteIntArray*)&inputs1, nullptr, const_cast<void*>(static_cast<const void*>(&opdata1)), nullptr, 45, },
  { (TfLiteIntArray*)&inputs2, (TfLiteIntArray*)&outputs2, (TfLiteIntArray*)&inputs2, nullptr, const_cast<void*>(static_cast<const void*>(&opdata2)), nullptr, 344, },
  { (TfLiteIntArray*)&inputs3, (TfLiteIntArray*)&outputs3, (TfLiteIntArray*)&inputs3, nullptr, const_cast<void*>(static_cast<const void*>(&opdata3)), nullptr, 45, },
  { (TfLiteIntArray*)&inputs4, (TfLiteIntArray*)&outputs4, (TfLiteIntArray*)&inputs4, nullptr, const_cast<void*>(static_cast<const void*>(&opdata4)), nullptr, 356, },
  { (TfLiteIntArray*)&inputs5, (TfLiteIntArray*)&outputs5, (TfLiteIntArray*)&inputs5, nullptr, const_cast<void*>(static_cast<const void*>(&opdata5)), nullptr, 45, },
  { (TfLiteIntArray*)&inputs6, (TfLiteIntArray*)&outputs6, (TfLiteIntArray*)&inputs6, nullptr, const_cast<void*>(static_cast<const void*>(&opdata6)), nullptr, 312, },
  { (TfLiteIntArray*)&inputs7, (TfLiteIntArray*)&outputs7, (TfLiteIntArray*)&inputs7, nullptr, const_cast<void*>(static_cast<const void*>(&opdata7)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs8, (TfLiteIntArray*)&outputs8, (TfLiteIntArray*)&inputs8, nullptr, const_cast<void*>(static_cast<const void*>(&opdata8)), nullptr, 356, },
  { (TfLiteIntArray*)&inputs9, (TfLiteIntArray*)&outputs9, (TfLiteIntArray*)&inputs9, nullptr, const_cast<void*>(static_cast<const void*>(&opdata9)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs10, (TfLiteIntArray*)&outputs10, (TfLiteIntArray*)&inputs10, nullptr, const_cast<void*>(static_cast<const void*>(&opdata10)), nullptr, 312, },
  { (TfLiteIntArray*)&inputs11, (TfLiteIntArray*)&outputs11, (TfLiteIntArray*)&inputs11, nullptr, const_cast<void*>(static_cast<const void*>(&opdata11)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs12, (TfLiteIntArray*)&outputs12, (TfLiteIntArray*)&inputs12, nullptr, const_cast<void*>(static_cast<const void*>(&opdata12)), nullptr, 356, },
  { (TfLiteIntArray*)&inputs13, (TfLiteIntArray*)&outputs13, (TfLiteIntArray*)&inputs13, nullptr, const_cast<void*>(static_cast<const void*>(&opdata13)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs14, (TfLiteIntArray*)&outputs14, (TfLiteIntArray*)&inputs14, nullptr, const_cast<void*>(static_cast<const void*>(&opdata14)), nullptr, 304, },
  { (TfLiteIntArray*)&inputs15, (TfLiteIntArray*)&outputs15, (TfLiteIntArray*)&inputs15, nullptr, const_cast<void*>(static_cast<const void*>(&opdata15)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs16, (TfLiteIntArray*)&outputs16, (TfLiteIntArray*)&inputs16, nullptr, const_cast<void*>(static_cast<const void*>(&opdata16)), nullptr, 356, },
  { (TfLiteIntArray*)&inputs17, (TfLiteIntArray*)&outputs17, (TfLiteIntArray*)&inputs17, nullptr, const_cast<void*>(static_cast<const void*>(&opdata17)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs18, (TfLiteIntArray*)&outputs18, (TfLiteIntArray*)&inputs18, nullptr, const_cast<void*>(static_cast<const void*>(&opdata18)), nullptr, 304, },
  { (TfLiteIntArray*)&inputs19, (TfLiteIntArray*)&outputs19, (TfLiteIntArray*)&inputs19, nullptr, const_cast<void*>(static_cast<const void*>(&opdata19)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs20, (TfLiteIntArray*)&outputs20, (TfLiteIntArray*)&inputs20, nullptr, const_cast<void*>(static_cast<const void*>(&opdata20)), nullptr, 356, },
  { (TfLiteIntArray*)&inputs21, (TfLiteIntArray*)&outputs21, (TfLiteIntArray*)&inputs21, nullptr, const_cast<void*>(static_cast<const void*>(&opdata21)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs22, (TfLiteIntArray*)&outputs22, (TfLiteIntArray*)&inputs22, nullptr, const_cast<void*>(static_cast<const void*>(&opdata22)), nullptr, 304, },
  { (TfLiteIntArray*)&inputs23, (TfLiteIntArray*)&outputs23, (TfLiteIntArray*)&inputs23, nullptr, const_cast<void*>(static_cast<const void*>(&opdata23)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs24, (TfLiteIntArray*)&outputs24, (TfLiteIntArray*)&inputs24, nullptr, const_cast<void*>(static_cast<const void*>(&opdata24)), nullptr, 356, },
  { (TfLiteIntArray*)&inputs25, (TfLiteIntArray*)&outputs25, (TfLiteIntArray*)&inputs25, nullptr, const_cast<void*>(static_cast<const void*>(&opdata25)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs26, (TfLiteIntArray*)&outputs26, (TfLiteIntArray*)&inputs26, nullptr, const_cast<void*>(static_cast<const void*>(&opdata26)), nullptr, 304, },
  { (TfLiteIntArray*)&inputs27, (TfLiteIntArray*)&outputs27, (TfLiteIntArray*)&inputs27, nullptr, const_cast<void*>(static_cast<const void*>(&opdata27)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs28, (TfLiteIntArray*)&outputs28, (TfLiteIntArray*)&inputs28, nullptr, const_cast<void*>(static_cast<const void*>(&opdata28)), nullptr, 356, },
  { (TfLiteIntArray*)&inputs29, (TfLiteIntArray*)&outputs29, (TfLiteIntArray*)&inputs29, nullptr, const_cast<void*>(static_cast<const void*>(&opdata29)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs30, (TfLiteIntArray*)&outputs30, (TfLiteIntArray*)&inputs30, nullptr, const_cast<void*>(static_cast<const void*>(&opdata30)), nullptr, 304, },
  { (TfLiteIntArray*)&inputs31, (TfLiteIntArray*)&outputs31, (TfLiteIntArray*)&inputs31, nullptr, const_cast<void*>(static_cast<const void*>(&opdata31)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs32, (TfLiteIntArray*)&outputs32, (TfLiteIntArray*)&inputs32, nullptr, const_cast<void*>(static_cast<const void*>(&opdata32)), nullptr, 356, },
  { (TfLiteIntArray*)&inputs33, (TfLiteIntArray*)&outputs33, (TfLiteIntArray*)&inputs33, nullptr, const_cast<void*>(static_cast<const void*>(&opdata33)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs34, (TfLiteIntArray*)&outputs34, (TfLiteIntArray*)&inputs34, nullptr, const_cast<void*>(static_cast<const void*>(&opdata34)), nullptr, 304, },
  { (TfLiteIntArray*)&inputs35, (TfLiteIntArray*)&outputs35, (TfLiteIntArray*)&inputs35, nullptr, const_cast<void*>(static_cast<const void*>(&opdata35)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs36, (TfLiteIntArray*)&outputs36, (TfLiteIntArray*)&inputs36, nullptr, const_cast<void*>(static_cast<const void*>(&opdata36)), nullptr, 356, },
  { (TfLiteIntArray*)&inputs37, (TfLiteIntArray*)&outputs37, (TfLiteIntArray*)&inputs37, nullptr, const_cast<void*>(static_cast<const void*>(&opdata37)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs38, (TfLiteIntArray*)&outputs38, (TfLiteIntArray*)&inputs38, nullptr, const_cast<void*>(static_cast<const void*>(&opdata38)), nullptr, 304, },
  { (TfLiteIntArray*)&inputs39, (TfLiteIntArray*)&outputs39, (TfLiteIntArray*)&inputs39, nullptr, const_cast<void*>(static_cast<const void*>(&opdata39)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs40, (TfLiteIntArray*)&outputs40, (TfLiteIntArray*)&inputs40, nullptr, const_cast<void*>(static_cast<const void*>(&opdata40)), nullptr, 356, },
  { (TfLiteIntArray*)&inputs41, (TfLiteIntArray*)&outputs41, (TfLiteIntArray*)&inputs41, nullptr, const_cast<void*>(static_cast<const void*>(&opdata41)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs42, (TfLiteIntArray*)&outputs42, (TfLiteIntArray*)&inputs42, nullptr, const_cast<void*>(static_cast<const void*>(&opdata42)), nullptr, 304, },
  { (TfLiteIntArray*)&inputs43, (TfLiteIntArray*)&outputs43, (TfLiteIntArray*)&inputs43, nullptr, const_cast<void*>(static_cast<const void*>(&opdata43)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs44, (TfLiteIntArray*)&outputs44, (TfLiteIntArray*)&inputs44, nullptr, const_cast<void*>(static_cast<const void*>(&opdata44)), nullptr, 356, },
  { (TfLiteIntArray*)&inputs45, (TfLiteIntArray*)&outputs45, (TfLiteIntArray*)&inputs45, nullptr, const_cast<void*>(static_cast<const void*>(&opdata45)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs46, (TfLiteIntArray*)&outputs46, (TfLiteIntArray*)&inputs46, nullptr, const_cast<void*>(static_cast<const void*>(&opdata46)), nullptr, 304, },
  { (TfLiteIntArray*)&inputs47, (TfLiteIntArray*)&outputs47, (TfLiteIntArray*)&inputs47, nullptr, const_cast<void*>(static_cast<const void*>(&opdata47)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs48, (TfLiteIntArray*)&outputs48, (TfLiteIntArray*)&inputs48, nullptr, const_cast<void*>(static_cast<const void*>(&opdata48)), nullptr, 356, },
  { (TfLiteIntArray*)&inputs49, (TfLiteIntArray*)&outputs49, (TfLiteIntArray*)&inputs49, nullptr, const_cast<void*>(static_cast<const void*>(&opdata49)), nullptr, 57, },
  { (TfLiteIntArray*)&inputs50, (TfLiteIntArray*)&outputs50, (TfLiteIntArray*)&inputs50, nullptr, const_cast<void*>(static_cast<const void*>(&opdata50)), nullptr, 304, },
  { (TfLiteIntArray*)&inputs51, (TfLiteIntArray*)&outputs51, (TfLiteIntArray*)&inputs51, nullptr, const_cast<void*>(static_cast<const void*>(&opdata51)), nullptr, 49, },
  { (TfLiteIntArray*)&inputs52, (TfLiteIntArray*)&outputs52, (TfLiteIntArray*)&inputs52, nullptr, const_cast<void*>(static_cast<const void*>(&opdata52)), nullptr, 356, },
  { (TfLiteIntArray*)&inputs53, (TfLiteIntArray*)&outputs53, (TfLiteIntArray*)&inputs53, nullptr, const_cast<void*>(static_cast<const void*>(&opdata53)), nullptr, 57, },
  { (TfLiteIntArray*)&inputs54, (TfLiteIntArray*)&outputs54, (TfLiteIntArray*)&inputs54, nullptr, const_cast<void*>(static_cast<const void*>(&opdata54)), nullptr, 304, },
  { (TfLiteIntArray*)&inputs55, (TfLiteIntArray*)&outputs55, (TfLiteIntArray*)&inputs55, nullptr, const_cast<void*>(static_cast<const void*>(&opdata55)), nullptr, 0, },
  { (TfLiteIntArray*)&inputs56, (TfLiteIntArray*)&outputs56, (TfLiteIntArray*)&inputs56, nullptr, const_cast<void*>(static_cast<const void*>(&opdata56)), nullptr, 39, },
  { (TfLiteIntArray*)&inputs57, (TfLiteIntArray*)&outputs57, (TfLiteIntArray*)&inputs57, nullptr, const_cast<void*>(static_cast<const void*>(&opdata57)), nullptr, 160, },
  { (TfLiteIntArray*)&inputs58, (TfLiteIntArray*)&outputs58, (TfLiteIntArray*)&inputs58, nullptr, const_cast<void*>(static_cast<const void*>(&opdata58)), nullptr, 0, },
};
used_operators_e used_ops[] = {
OP_PAD, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_MEAN, OP_XC_ld_flash, OP_XC_conv2d_v2, OP_SOFTMAX, };


// Scratch buffer variables
int scratch_buffer_idx = 0;
const int scratch_buffer_offsets[137] = { 201856, 201760, 201664, 201568, 201472, 704, 528, 352, 176, 0, 301824, 301760, 301696, 301632, 301568, 251584, 251408, 251232, 251056, 250880, 101504, 101440, 101376, 101312, 101248, 201904, 201728, 201552, 201376, 201200, 0, 0, 0, 0, 0, 126448, 126272, 126096, 125920, 125744, 0, 0, 0, 0, 0, 101904, 101728, 101552, 101376, 101200, 0, 0, 0, 0, 0, 64272, 64096, 63920, 63744, 63568, 0, 0, 0, 0, 0, 52560, 52384, 52208, 52032, 51856, 0, 0, 0, 0, 0, 52560, 52384, 52208, 52032, 51856, 0, 0, 0, 0, 0, 52560, 52384, 52208, 52032, 51856, 0, 0, 0, 0, 0, 52560, 52384, 52208, 52032, 51856, 0, 0, 0, 0, 0, 52560, 52384, 52208, 52032, 51856, 0, 0, 0, 0, 0, 27472, 27296, 27120, 26944, 26768, 0, 0, 0, 0, 0, 4560, 4384, 4208, 4032, 3856, 0, 0, 0, 0, 0, 0, 261280 };
tflite::MicroContext mc;

// Xcore context and thread variables
xc_context_config_t xc_config;
constexpr int kStackWordsPerThread = 256;
constexpr int threadsStackSizeInUint64 = 5 * kStackWordsPerThread/2;
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
  ctx.tensors_size = 90;
  registrations[OP_PAD] = tflite::ops::micro::Register_PAD();
  registrations[OP_XC_ld_flash] = *(tflite::ops::micro::xcore::Register_XC_ld_flash());
  registrations[OP_XC_conv2d_v2] = *(tflite::ops::micro::xcore::Register_XC_conv2d_v2());
  registrations[OP_MEAN] = tflite::Register_MEAN();
  registrations[OP_SOFTMAX] = tflite::Register_SOFTMAX();


#ifdef TFLMC_XCORE_PROFILE
  printf("\nProfiling init()...");
  memset(op_times, 0, sizeof(op_times));
#endif

  for(size_t i = 0; i < 59; ++i) {
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

  for(size_t i = 0; i < 59; ++i) {
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
  89, 
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
  thread_init_5(&xc_config.thread_info);

#ifdef TFLMC_XCORE_PROFILE
  printf("\nProfiling invoke()...");
  memset(op_times, 0, sizeof(op_times));
  memset(op_counts, 0, sizeof(op_counts));
#endif

  for(size_t i = 0; i < 59; ++i) {

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
      printf("\noutput %d, %d bytes, checksum %d\n", j, tflTensors[tflNodes[i].outputs->data[j]].bytes, checksum(tflTensors[tflNodes[i].outputs->data[j]].data.raw, tflTensors[tflNodes[i].outputs->data[j]].bytes));
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
  for(size_t i = 0; i < 59; ++i) {
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
    printf("\n%-5d %-32s %-12d", op_counts[i], op_strs[i], op_times[i]);
  }
  printf("\n");
#endif

  return kTfLiteOk;
}
