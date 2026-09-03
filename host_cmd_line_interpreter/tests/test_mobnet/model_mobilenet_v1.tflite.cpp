// This file is generated. Do not edit.
// Generated on: 03.09.2026 12:40:55
// Compiler version: Not_built_with_version_info!
// Args: mobilenet_v1_0.25_128.tflite --xcore-weights-file=model_mobilenet_v1.params -o model_mobilenet_v1.tflite 

#include "lib_tflite_micro/api/xcore_config.h"
#include "lib_nn/api/version.h"
#include "lib_nn/api/nn_arch.h"
#include "lib_tflite_micro/api/version.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/conv.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/kernels/reduce.h"
#include "tensorflow/lite/micro/kernels/softmax.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/memory_helpers.h"

#if defined(__xcore__) || defined(__riscv_xxcore)
#include <xcore/hwtimer.h>
#endif

// #define TFLMC_XCORE_PROFILE
// #define TFLMC_CONV2D_PROFILE
// #define TFLMC_PRINT_TENSORS
// #define TFLMC_PRINT_INPUT_TENSORS

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

// Check target arch
#ifdef __XS3A__
static_assert(0 == 0, "Model has not been compiled for XS3A!");
#elif __VX4A__ || __VX4B__
static_assert(0 == 1, "Model has not been compiled for VX4A!");
#endif

// Check lib_nn and lib_tflite_micro versions
// NOTE: xformer version is saved for debugging purposes
// If lib_nn and lib_tflite_micro versions are as expected,
// then the xformer version doesn't matter as the model should execute
// If major version is zero, then minor versions must match
// Otherwise, major versions must match and binary minor version
// must be less or equal to runtime minor version
// Check if runtime lib_tflite_micro version matches with compiled version
static_assert((0 == 0 && lib_tflite_micro::major_version == 0 && 8 == lib_tflite_micro::minor_version) ||
              (0 == lib_tflite_micro::major_version) ||
              (8  < lib_tflite_micro::minor_version),
             "Model has been compiled with lib_tflite_micro version incompatible with runtime lib_tflite_micro version!");

// Check if runtime lib_nn version matches with compiled version
static_assert((0 == 0 && lib_nn::major_version == 0 && 6 == lib_nn::minor_version) ||
              (0 == lib_nn::major_version) ||
              (6  < lib_nn::minor_version),
             "Model has been compiled with lib_nn version incompatible with runtime lib_nn version!");

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
extern TFLMRegistration *Register_XC_pad_3_to_4(void);
extern TFLMRegistration *Register_XC_pad(void);
extern TFLMRegistration *Register_XC_ld_weights(void);
extern TFLMRegistration *Register_XC_conv2d_v2(void);
extern TFLMRegistration *Register_XC_lookup(void);
extern TFLMRegistration *Register_XC_mean(void);
extern TFLMRegistration *Register_XC_slice(void);
extern TFLMRegistration *Register_XC_no_op(void);
extern TFLMRegistration *Register_XC_softmax(void);
} // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite_micro



constexpr int kTensorArenaSize = 139648;
#ifndef SHARED_TENSOR_ARENA
namespace {
uint8_t tensor_arena[kTensorArenaSize] ALIGN(8);
}
#else
extern uint8_t tensor_arena[];
#endif

namespace {
template <int SZ, class T> struct TfArray {
  int sz; T elem[SZ];
};
enum used_operators_e {
  OP_XC_pad_3_to_4, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_mean, OP_XC_slice, OP_XC_no_op, OP_XC_softmax,  OP_LAST
};

#if defined(TFLMC_XCORE_PROFILE) || defined(TFLMC_PRINT_TENSORS) || defined(TFLMC_PRINT_INPUT_TENSORS) || defined(TFLMC_CONV2D_PROFILE)
const char *op_strs[] = {
"OP_XC_pad_3_to_4", "OP_XC_pad", "OP_XC_ld_weights", "OP_XC_conv2d_v2", "OP_XC_lookup", "OP_XC_mean", "OP_XC_slice", "OP_XC_no_op", "OP_XC_softmax", };

#endif
#if defined(TFLMC_XCORE_PROFILE) || defined(TFLMC_PRINT_TENSORS) || defined(TFLMC_PRINT_INPUT_TENSORS)
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

#ifdef TFLMC_XCORE_PROFILE
int op_times[OP_LAST];
int op_counts[OP_LAST];
int64_t op_times_summed;
int time_t0, time_t1;
#endif

TfLiteContext ctx{};

TFLMRegistration registrations[OP_LAST];

struct {
const TfArray<4, int> tensor_dimension0 = { 4, { 1,128,128,3 } };
const TfArray<1, float> quant0_scale = { 1, { 0.00039607842336408794, } };
const TfArray<1, int> quant0_zero = { 1, { 124 } };
const TfLiteAffineQuantization quant0 = { (TfLiteFloatArray*)&quant0_scale, (TfLiteIntArray*)&quant0_zero, 0 };
const ALIGN(8) int16_t tensor_data1[28] = { 
    9858, 9915, 9917, 9904, 9914, 9881, 9919, 9854, 9855, 9806, 
    0, 0, -1, -1, -1, -1, -1, -1, -1, -1, 
    -1, -1, -1, -1, 0, 0, 0, 0, 
};
const TfArray<1, int> tensor_dimension1 = { 1, { 28 } };
const ALIGN(8) int16_t tensor_data2[32] = { 
    11612, 11966, 10654, 10640, 11198, 11766, 10301, 11314, 9745, 8503, 
    9147, 8490, 11864, 11927, 9766, 10321, -4709, -17535, -12295, 4714, 
    -4513, -3821, -5847, 5626, -5525, -1573, 834, -333, -714, -9897, 
    -1465, -6234, 
};
const TfArray<1, int> tensor_dimension2 = { 1, { 32 } };
const ALIGN(8) int16_t tensor_data3[32] = { 
    22869, 19842, 20858, 19019, 25553, 25045, 24834, 24065, 23301, 24243, 
    22426, 23978, 22276, 23191, 21458, 25396, -9890, -18075, -11452, -7784, 
    -5947, -4965, -5137, -7770, 5233, -23818, 10291, -138, -13849, -4433, 
    -16407, -21684, 
};
const ALIGN(8) int16_t tensor_data4[24] = { 
    26633, 26866, 23217, 23576, 21167, 27696, 0, 27395, -10898, -23935, 
    -6878, -6858, -15511, -15225, -8193, -9263, 0, 0, 0, 0, 
    0, 0, 0, 0, 
};
const TfArray<1, int> tensor_dimension4 = { 1, { 24 } };
const ALIGN(8) int16_t tensor_data5[24] = { 
    11308, 11475, 11627, 11351, 11425, 10720, 10111, 11483, -9348, -7205, 
    -4431, -6410, -10765, -3346, -12392, -13484, 0, 0, 0, 0, 
    0, 0, 0, 0, 
};
const TfArray<4, int> tensor_dimension6 = { 4, { 1,128,128,4 } };
const TfArray<4, int> tensor_dimension7 = { 4, { 1,129,129,4 } };
const TfArray<1, int> tensor_dimension8 = { 1, { 96 } };
const TfArray<1, int> tensor_dimension9 = { 1, { 768 } };
const TfArray<4, int> tensor_dimension10 = { 4, { 1,64,64,8 } };
const TfArray<1, float> quant10_scale = { 1, { 0.00055054255062714219, } };
const TfArray<1, int> quant10_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant10 = { (TfLiteFloatArray*)&quant10_scale, (TfLiteIntArray*)&quant10_zero, 0 };
const TfArray<4, int> tensor_dimension11 = { 4, { 1,66,66,8 } };
const TfArray<1, int> tensor_dimension12 = { 1, { 160 } };
const TfArray<1, float> quant13_scale = { 1, { 0.0001772949326550588, } };
const TfArray<1, int> quant13_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant13 = { (TfLiteFloatArray*)&quant13_scale, (TfLiteIntArray*)&quant13_zero, 0 };
const TfArray<1, int> tensor_dimension14 = { 1, { 64 } };
const TfArray<1, int> tensor_dimension15 = { 1, { 512 } };
const TfArray<4, int> tensor_dimension16 = { 4, { 1,64,64,16 } };
const TfArray<1, float> quant16_scale = { 1, { 0.00011377604823792353, } };
const TfArray<1, int> quant16_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant16 = { (TfLiteFloatArray*)&quant16_scale, (TfLiteIntArray*)&quant16_zero, 0 };
const TfArray<4, int> tensor_dimension17 = { 4, { 1,65,65,16 } };
const TfArray<4, int> tensor_dimension19 = { 4, { 1,32,32,16 } };
const TfArray<1, float> quant19_scale = { 1, { 3.0618419259553775e-05, } };
const TfArray<1, int> quant19_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant19 = { (TfLiteFloatArray*)&quant19_scale, (TfLiteIntArray*)&quant19_zero, 0 };
const TfArray<4, int> tensor_dimension23 = { 4, { 1,32,32,32 } };
const TfArray<1, float> quant23_scale = { 1, { 1.0048755029856693e-05, } };
const TfArray<1, int> quant23_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant23 = { (TfLiteFloatArray*)&quant23_scale, (TfLiteIntArray*)&quant23_zero, 0 };
const TfArray<4, int> tensor_dimension24 = { 4, { 1,34,34,32 } };
const TfArray<1, int> tensor_dimension25 = { 1, { 304 } };
const TfArray<1, float> quant27_scale = { 1, { 2.5023409762070514e-06, } };
const TfArray<1, int> quant27_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant27 = { (TfLiteFloatArray*)&quant27_scale, (TfLiteIntArray*)&quant27_zero, 0 };
const TfArray<1, int> tensor_dimension28 = { 1, { 1024 } };
const TfArray<1, float> quant30_scale = { 1, { 9.9999999747524271e-07, } };
const TfArray<1, int> quant30_zero = { 1, { 0 } };
const TfLiteAffineQuantization quant30 = { (TfLiteFloatArray*)&quant30_scale, (TfLiteIntArray*)&quant30_zero, 0 };
const TfArray<1, int> tensor_dimension31 = { 1, { 256 } };
const TfArray<4, int> tensor_dimension33 = { 4, { 1,33,33,32 } };
const TfArray<4, int> tensor_dimension36 = { 4, { 1,16,16,32 } };
const TfArray<1, int> tensor_dimension38 = { 1, { 2048 } };
const TfArray<1, int> tensor_dimension39 = { 1, { 128 } };
const TfArray<4, int> tensor_dimension40 = { 4, { 1,16,16,64 } };
const TfArray<4, int> tensor_dimension42 = { 4, { 1,18,18,64 } };
const TfArray<1, int> tensor_dimension43 = { 1, { 592 } };
const TfArray<1, int> tensor_dimension47 = { 1, { 4096 } };
const TfArray<4, int> tensor_dimension51 = { 4, { 1,17,17,64 } };
const TfArray<4, int> tensor_dimension54 = { 4, { 1,8,8,64 } };
const TfArray<1, int> tensor_dimension56 = { 1, { 8192 } };
const TfArray<4, int> tensor_dimension58 = { 4, { 1,8,8,128 } };
const TfArray<4, int> tensor_dimension60 = { 4, { 1,10,10,128 } };
const TfArray<1, int> tensor_dimension61 = { 1, { 1168 } };
const TfArray<1, int> tensor_dimension65 = { 1, { 16384 } };
const TfArray<4, int> tensor_dimension105 = { 4, { 1,9,9,128 } };
const TfArray<4, int> tensor_dimension108 = { 4, { 1,4,4,128 } };
const TfArray<1, int> tensor_dimension110 = { 1, { 32768 } };
const TfArray<4, int> tensor_dimension112 = { 4, { 1,4,4,256 } };
const TfArray<4, int> tensor_dimension114 = { 4, { 1,6,6,256 } };
const TfArray<1, int> tensor_dimension115 = { 1, { 2320 } };
const TfArray<1, int> tensor_dimension119 = { 1, { 65536 } };
const TfArray<2, int> tensor_dimension123 = { 2, { 1,256 } };
const TfArray<1, int> tensor_dimension124 = { 1, { 288 } };
const TfArray<1, int> tensor_dimension125 = { 1, { 3200 } };
const TfArray<4, int> tensor_dimension126 = { 4, { 1,1,1,12 } };
const TfArray<4, int> tensor_dimension127 = { 4, { 1,1,1,10 } };
const TfArray<2, int> tensor_dimension128 = { 2, { 1,10 } };
const TfArray<1, float> quant130_scale = { 1, { 0.00390625, } };
const TfArray<1, int> quant130_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant130 = { (TfLiteFloatArray*)&quant130_scale, (TfLiteIntArray*)&quant130_zero, 0 };
uint8_t ALIGN(4) opdata0[28] = { 112, 118, 0, 1, 4, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 124, 124, 124, 124, 6, 5, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs0 = { 1, { 0 } };
const TfArray<1, int> outputs0 = { 1, { 6 } };
uint8_t ALIGN(4) opdata1[74] = { 115, 0, 112, 0, 108, 0, 110, 0, 122, 0, 101, 0, 118, 0, 7, 5, 12, 11, 16, 19, 8, 13, 0, 0, 9, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 8, 2, 0, 0, 0, 2, 0, 0, 127, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 124, 124, 124, 124, 6, 6, 6, 6, 6, 106, 6, 35, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs1 = { 1, { 6 } };
const TfArray<1, int> outputs1 = { 1, { 7 } };
uint8_t ALIGN(4) opdata2[46] = { 97, 0, 115, 0, 1, 0, 0, 3, 5, 116, 0, 3, 12, 11, 5, 0, 4, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 224, 100, 3, 0, 26, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs2 = 0; /* empty TfLiteIntArray */
const TfArray<1, int> outputs2 = { 1, { 9 } };
uint8_t ALIGN(4) opdata3[152] = { 109, 112, 0, 40, 8, 4, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 252, 255, 255, 255, 228, 255, 255, 255, 248, 1, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 97, 0, 8, 8, 0, 0, 0, 36, 0, 0, 0, 0, 111, 0, 8, 8, 0, 0, 0, 3, 0, 254, 255, 0, 112, 0, 42, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 80, 6, 127, 71, 60, 12, 9, 7, 1, 7, 87, 1, 133, 78, 23, 96, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs3 = { 5, { 7,9,5,-1,8 } };
const TfArray<1, int> outputs3 = { 1, { 10 } };
uint8_t ALIGN(4) opdata4[74] = { 115, 0, 112, 0, 108, 0, 110, 0, 122, 0, 101, 0, 118, 0, 7, 5, 12, 11, 16, 19, 8, 13, 0, 0, 9, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 24, 2, 0, 0, 0, 2, 0, 0, 63, 0, 0, 0, 16, 0, 0, 0, 24, 2, 0, 0, 1, 0, 0, 0, 128, 128, 128, 128, 6, 6, 6, 6, 6, 106, 6, 35, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs4 = { 1, { 10 } };
const TfArray<1, int> outputs4 = { 1, { 11 } };
uint8_t ALIGN(4) opdata5[46] = { 97, 0, 115, 0, 1, 0, 160, 0, 5, 116, 0, 3, 12, 11, 5, 0, 4, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 64, 100, 3, 0, 26, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs5 = 0; /* empty TfLiteIntArray */
const TfArray<1, int> outputs5 = { 1, { 12 } };
uint8_t ALIGN(4) opdata6[132] = { 109, 112, 0, 8, 16, 2, 0, 0, 8, 0, 0, 0, 0, 97, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 248, 1, 0, 0, 0, 111, 0, 8, 8, 0, 0, 0, 2, 0, 254, 255, 0, 112, 0, 42, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 92, 6, 107, 71, 60, 12, 9, 7, 1, 7, 99, 3, 113, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs6 = { 5, { 11,12,4,-1,-1 } };
const TfArray<1, int> outputs6 = { 1, { 13 } };
uint8_t ALIGN(4) opdata7[46] = { 97, 0, 115, 0, 1, 0, 0, 2, 5, 116, 0, 3, 12, 11, 5, 0, 4, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 64, 98, 3, 0, 26, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs7 = 0; /* empty TfLiteIntArray */
const TfArray<1, int> outputs7 = { 1, { 15 } };
uint8_t ALIGN(4) opdata8[152] = { 109, 112, 0, 40, 0, 2, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 255, 255, 255, 232, 255, 255, 255, 248, 1, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 97, 0, 8, 16, 0, 0, 0, 8, 0, 0, 0, 0, 111, 0, 8, 16, 0, 0, 0, 2, 0, 254, 255, 0, 112, 0, 42, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 80, 6, 127, 71, 60, 12, 9, 7, 1, 7, 87, 1, 133, 78, 23, 64, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs8 = { 5, { 13,15,3,-1,14 } };
const TfArray<1, int> outputs8 = { 1, { 16 } };
uint8_t ALIGN(4) opdata9[74] = { 115, 0, 112, 0, 108, 0, 110, 0, 122, 0, 101, 0, 118, 0, 7, 5, 12, 11, 16, 19, 8, 13, 0, 0, 9, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 32, 4, 0, 0, 0, 4, 0, 0, 63, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 128, 128, 128, 128, 6, 6, 6, 6, 6, 106, 6, 35, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs9 = { 1, { 16 } };
const TfArray<1, int> outputs9 = { 1, { 17 } };
uint8_t ALIGN(4) opdata10[46] = { 97, 0, 115, 0, 1, 0, 160, 0, 5, 116, 0, 3, 12, 11, 5, 0, 4, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 160, 97, 3, 0, 26, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs10 = 0; /* empty TfLiteIntArray */
const TfArray<1, int> outputs10 = { 1, { 18 } };
uint8_t ALIGN(4) opdata11[132] = { 109, 112, 0, 8, 32, 8, 0, 0, 32, 0, 0, 0, 0, 97, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 16, 0, 0, 0, 224, 3, 0, 0, 0, 111, 0, 8, 16, 0, 0, 0, 2, 0, 253, 255, 0, 112, 0, 42, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 92, 6, 107, 71, 60, 12, 9, 7, 1, 7, 99, 3, 113, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs11 = { 5, { 17,18,2,-1,-1 } };
const TfArray<1, int> outputs11 = { 1, { 19 } };
uint8_t ALIGN(4) opdata12[50] = { 97, 0, 115, 0, 2, 0, 0, 3, 128, 0, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 32, 94, 3, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs12 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs12 = { 2, { 21,22 } };
uint8_t ALIGN(4) opdata13[152] = { 109, 112, 0, 40, 0, 2, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 255, 255, 255, 240, 255, 255, 255, 240, 1, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 97, 0, 8, 32, 0, 0, 0, 16, 0, 0, 0, 0, 111, 0, 8, 32, 0, 0, 0, 2, 0, 253, 255, 0, 112, 0, 42, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 80, 6, 127, 71, 60, 12, 9, 7, 1, 7, 87, 1, 133, 78, 23, 64, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs13 = { 5, { 19,21,22,-1,20 } };
const TfArray<1, int> outputs13 = { 1, { 23 } };
uint8_t ALIGN(4) opdata14[74] = { 115, 0, 112, 0, 108, 0, 110, 0, 122, 0, 101, 0, 118, 0, 7, 5, 12, 11, 16, 19, 8, 13, 0, 0, 9, 0, 0, 0, 1, 0, 0, 0, 7, 0, 0, 0, 96, 4, 0, 0, 0, 4, 0, 0, 31, 0, 0, 0, 64, 0, 0, 0, 96, 4, 0, 0, 1, 0, 0, 0, 128, 128, 128, 128, 6, 6, 6, 6, 6, 106, 6, 35, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs14 = { 1, { 23 } };
const TfArray<1, int> outputs14 = { 1, { 24 } };
uint8_t ALIGN(4) opdata15[50] = { 97, 0, 115, 0, 2, 0, 48, 1, 128, 0, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 96, 92, 3, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs15 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs15 = { 2, { 25,26 } };
uint8_t ALIGN(4) opdata16[132] = { 109, 112, 0, 8, 64, 4, 0, 0, 32, 0, 0, 0, 0, 97, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 32, 0, 0, 0, 224, 3, 0, 0, 0, 111, 0, 8, 32, 0, 0, 0, 2, 0, 254, 255, 0, 112, 0, 42, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 92, 6, 107, 71, 60, 12, 9, 7, 1, 7, 99, 3, 113, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs16 = { 5, { 24,25,26,-1,-1 } };
const TfArray<1, int> outputs16 = { 1, { 27 } };
uint8_t ALIGN(4) opdata17[50] = { 97, 0, 115, 0, 2, 0, 0, 4, 128, 0, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 224, 87, 3, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs17 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs17 = { 2, { 28,29 } };
uint8_t ALIGN(4) opdata18[136] = { 109, 112, 0, 8, 0, 4, 0, 0, 32, 0, 0, 0, 0, 97, 0, 24, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 224, 3, 0, 0, 0, 111, 0, 8, 32, 0, 0, 0, 2, 0, 253, 255, 0, 112, 0, 42, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 96, 6, 111, 71, 60, 12, 9, 7, 1, 7, 103, 0, 117, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs18 = { 5, { 27,28,29,-1,-1 } };
const TfArray<1, int> outputs18 = { 1, { 30 } };
uint8_t ALIGN(4) opdata19[46] = { 97, 0, 115, 0, 1, 0, 0, 1, 5, 116, 0, 3, 12, 11, 5, 0, 4, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 224, 107, 3, 0, 26, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs19 = 0; /* empty TfLiteIntArray */
const TfArray<1, int> outputs19 = { 1, { 31 } };
uint8_t ALIGN(4) opdata20[0] = {  }; /* custom_initial_data */
const TfArray<2, int> inputs20 = { 2, { 30,31 } };
const TfArray<1, int> outputs20 = { 1, { 32 } };
uint8_t ALIGN(4) opdata21[52] = { 115, 0, 112, 0, 108, 0, 110, 0, 122, 0, 101, 0, 118, 0, 7, 5, 12, 11, 16, 19, 8, 13, 7, 0, 1, 0, 7, 0, 64, 4, 0, 4, 31, 0, 32, 0, 0, 0, 1, 0, 0, 0, 5, 5, 5, 5, 5, 105, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs21 = { 1, { 32 } };
const TfArray<1, int> outputs21 = { 1, { 33 } };
uint8_t ALIGN(4) opdata22[50] = { 97, 0, 115, 0, 2, 0, 48, 1, 128, 0, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 32, 86, 3, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs22 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs22 = { 2, { 34,35 } };
uint8_t ALIGN(4) opdata23[132] = { 109, 112, 0, 8, 64, 8, 0, 0, 64, 0, 0, 0, 0, 97, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 32, 0, 0, 0, 192, 3, 0, 0, 0, 111, 0, 8, 32, 0, 0, 0, 2, 0, 0, 0, 0, 112, 0, 42, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 92, 6, 107, 71, 60, 12, 9, 7, 1, 7, 99, 3, 113, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs23 = { 5, { 33,34,35,-1,-1 } };
const TfArray<1, int> outputs23 = { 1, { 36 } };
const TfArray<2, int> inputs24 = { 2, { 36,31 } };
const TfArray<1, int> outputs24 = { 1, { 37 } };
uint8_t ALIGN(4) opdata25[50] = { 97, 0, 115, 0, 2, 0, 0, 8, 0, 1, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 32, 77, 3, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs25 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs25 = { 2, { 38,39 } };
uint8_t ALIGN(4) opdata26[136] = { 109, 112, 0, 8, 0, 2, 0, 0, 32, 0, 0, 0, 0, 97, 0, 24, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 224, 1, 0, 0, 0, 111, 0, 8, 64, 0, 0, 0, 2, 0, 255, 255, 0, 112, 0, 42, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 96, 6, 111, 71, 60, 12, 9, 7, 1, 7, 103, 0, 117, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs26 = { 5, { 37,38,39,-1,-1 } };
const TfArray<1, int> outputs26 = { 1, { 40 } };
const TfArray<2, int> inputs27 = { 2, { 40,31 } };
const TfArray<1, int> outputs27 = { 1, { 41 } };
uint8_t ALIGN(4) opdata28[52] = { 115, 0, 112, 0, 108, 0, 110, 0, 122, 0, 101, 0, 118, 0, 7, 5, 12, 11, 16, 19, 8, 13, 7, 0, 1, 0, 7, 0, 192, 4, 0, 4, 15, 0, 128, 0, 192, 4, 1, 0, 0, 0, 5, 5, 5, 5, 5, 105, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs28 = { 1, { 41 } };
const TfArray<1, int> outputs28 = { 1, { 42 } };
uint8_t ALIGN(4) opdata29[50] = { 97, 0, 115, 0, 2, 0, 80, 2, 0, 1, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 192, 73, 3, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs29 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs29 = { 2, { 43,44 } };
uint8_t ALIGN(4) opdata30[132] = { 109, 112, 0, 8, 128, 4, 0, 0, 64, 0, 0, 0, 0, 97, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 64, 0, 0, 0, 192, 3, 0, 0, 0, 111, 0, 8, 64, 0, 0, 0, 2, 0, 0, 0, 0, 112, 0, 42, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 92, 6, 107, 71, 60, 12, 9, 7, 1, 7, 99, 3, 113, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs30 = { 5, { 42,43,44,-1,-1 } };
const TfArray<1, int> outputs30 = { 1, { 45 } };
const TfArray<2, int> inputs31 = { 2, { 45,31 } };
const TfArray<1, int> outputs31 = { 1, { 46 } };
uint8_t ALIGN(4) opdata32[50] = { 97, 0, 115, 0, 2, 0, 0, 16, 0, 1, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 192, 56, 3, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs32 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs32 = { 2, { 47,48 } };
uint8_t ALIGN(4) opdata33[136] = { 109, 112, 0, 8, 0, 4, 0, 0, 64, 0, 0, 0, 0, 97, 0, 24, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 192, 3, 0, 0, 0, 111, 0, 8, 64, 0, 0, 0, 2, 0, 255, 255, 0, 112, 0, 42, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 96, 6, 111, 71, 60, 12, 9, 7, 1, 7, 103, 0, 117, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs33 = { 5, { 46,47,48,-1,-1 } };
const TfArray<1, int> outputs33 = { 1, { 49 } };
const TfArray<2, int> inputs34 = { 2, { 49,31 } };
const TfArray<1, int> outputs34 = { 1, { 50 } };
uint8_t ALIGN(4) opdata35[52] = { 115, 0, 112, 0, 108, 0, 110, 0, 122, 0, 101, 0, 118, 0, 7, 5, 12, 11, 16, 19, 8, 13, 7, 0, 1, 0, 7, 0, 128, 4, 0, 4, 15, 0, 64, 0, 0, 0, 1, 0, 0, 0, 5, 5, 5, 5, 5, 105, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs35 = { 1, { 50 } };
const TfArray<1, int> outputs35 = { 1, { 51 } };
uint8_t ALIGN(4) opdata36[50] = { 97, 0, 115, 0, 2, 0, 80, 2, 0, 1, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 96, 53, 3, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs36 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs36 = { 2, { 52,53 } };
uint8_t ALIGN(4) opdata37[132] = { 109, 112, 0, 8, 128, 8, 0, 0, 128, 0, 0, 0, 0, 97, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 64, 0, 0, 0, 128, 3, 0, 0, 0, 111, 0, 8, 64, 0, 0, 0, 2, 0, 0, 0, 0, 112, 0, 42, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 92, 6, 107, 71, 60, 12, 9, 7, 1, 7, 99, 3, 113, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs37 = { 5, { 51,52,53,-1,-1 } };
const TfArray<1, int> outputs37 = { 1, { 54 } };
const TfArray<2, int> inputs38 = { 2, { 54,31 } };
const TfArray<1, int> outputs38 = { 1, { 55 } };
uint8_t ALIGN(4) opdata39[50] = { 97, 0, 115, 0, 2, 0, 0, 32, 0, 2, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 96, 19, 3, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs39 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs39 = { 2, { 56,57 } };
uint8_t ALIGN(4) opdata40[136] = { 109, 112, 0, 8, 0, 2, 0, 0, 64, 0, 0, 0, 0, 97, 0, 24, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 192, 1, 0, 0, 0, 111, 0, 8, 128, 0, 0, 0, 2, 0, 255, 255, 0, 112, 0, 42, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 96, 6, 111, 71, 60, 12, 9, 7, 1, 7, 103, 0, 117, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs40 = { 5, { 55,56,57,-1,-1 } };
const TfArray<1, int> outputs40 = { 1, { 58 } };
const TfArray<2, int> inputs41 = { 2, { 58,31 } };
const TfArray<1, int> outputs41 = { 1, { 59 } };
uint8_t ALIGN(4) opdata42[52] = { 115, 0, 112, 0, 108, 0, 110, 0, 122, 0, 101, 0, 118, 0, 7, 5, 12, 11, 16, 19, 8, 13, 7, 0, 1, 0, 7, 0, 128, 5, 0, 4, 7, 0, 0, 1, 128, 5, 1, 0, 0, 0, 5, 5, 5, 5, 5, 105, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs42 = { 1, { 59 } };
const TfArray<1, int> outputs42 = { 1, { 60 } };
uint8_t ALIGN(4) opdata43[50] = { 97, 0, 115, 0, 2, 0, 144, 4, 0, 2, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 192, 12, 3, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs43 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs43 = { 2, { 61,62 } };
uint8_t ALIGN(4) opdata44[132] = { 109, 112, 0, 8, 0, 5, 0, 0, 128, 0, 0, 0, 0, 97, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 128, 0, 0, 0, 128, 3, 0, 0, 0, 111, 0, 8, 128, 0, 0, 0, 2, 0, 1, 0, 0, 112, 0, 42, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 92, 6, 107, 71, 60, 12, 9, 7, 1, 7, 99, 3, 113, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs44 = { 5, { 60,61,62,-1,-1 } };
const TfArray<1, int> outputs44 = { 1, { 63 } };
const TfArray<2, int> inputs45 = { 2, { 63,31 } };
const TfArray<1, int> outputs45 = { 1, { 64 } };
uint8_t ALIGN(4) opdata46[50] = { 97, 0, 115, 0, 2, 0, 0, 64, 0, 2, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 192, 202, 2, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs46 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs46 = { 2, { 65,66 } };
uint8_t ALIGN(4) opdata47[136] = { 109, 112, 0, 8, 0, 4, 0, 0, 128, 0, 0, 0, 0, 97, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 128, 3, 0, 0, 0, 111, 0, 8, 128, 0, 0, 0, 2, 0, 255, 255, 0, 112, 0, 42, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 96, 6, 111, 71, 60, 12, 9, 7, 1, 7, 103, 0, 117, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs47 = { 5, { 64,65,66,-1,-1 } };
const TfArray<1, int> outputs47 = { 1, { 67 } };
const TfArray<2, int> inputs48 = { 2, { 67,31 } };
const TfArray<1, int> outputs48 = { 1, { 68 } };
const TfArray<1, int> inputs49 = { 1, { 68 } };
const TfArray<1, int> outputs49 = { 1, { 69 } };
uint8_t ALIGN(4) opdata50[50] = { 97, 0, 115, 0, 2, 0, 144, 4, 0, 2, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 32, 196, 2, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs50 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs50 = { 2, { 70,71 } };
const TfArray<5, int> inputs51 = { 5, { 69,70,71,-1,-1 } };
const TfArray<1, int> outputs51 = { 1, { 72 } };
const TfArray<2, int> inputs52 = { 2, { 72,31 } };
const TfArray<1, int> outputs52 = { 1, { 73 } };
uint8_t ALIGN(4) opdata53[50] = { 97, 0, 115, 0, 2, 0, 0, 64, 0, 2, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 32, 130, 2, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs53 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs53 = { 2, { 74,75 } };
const TfArray<5, int> inputs54 = { 5, { 73,74,75,-1,-1 } };
const TfArray<1, int> outputs54 = { 1, { 76 } };
const TfArray<2, int> inputs55 = { 2, { 76,31 } };
const TfArray<1, int> outputs55 = { 1, { 77 } };
const TfArray<1, int> inputs56 = { 1, { 77 } };
const TfArray<1, int> outputs56 = { 1, { 78 } };
uint8_t ALIGN(4) opdata57[50] = { 97, 0, 115, 0, 2, 0, 144, 4, 0, 2, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 128, 123, 2, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs57 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs57 = { 2, { 79,80 } };
const TfArray<5, int> inputs58 = { 5, { 78,79,80,-1,-1 } };
const TfArray<1, int> outputs58 = { 1, { 81 } };
const TfArray<2, int> inputs59 = { 2, { 81,31 } };
const TfArray<1, int> outputs59 = { 1, { 82 } };
uint8_t ALIGN(4) opdata60[50] = { 97, 0, 115, 0, 2, 0, 0, 64, 0, 2, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 128, 57, 2, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs60 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs60 = { 2, { 83,84 } };
const TfArray<5, int> inputs61 = { 5, { 82,83,84,-1,-1 } };
const TfArray<1, int> outputs61 = { 1, { 85 } };
const TfArray<2, int> inputs62 = { 2, { 85,31 } };
const TfArray<1, int> outputs62 = { 1, { 86 } };
const TfArray<1, int> inputs63 = { 1, { 86 } };
const TfArray<1, int> outputs63 = { 1, { 87 } };
uint8_t ALIGN(4) opdata64[50] = { 97, 0, 115, 0, 2, 0, 144, 4, 0, 2, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 224, 50, 2, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs64 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs64 = { 2, { 88,89 } };
uint8_t ALIGN(4) opdata65[132] = { 109, 112, 0, 8, 0, 5, 0, 0, 128, 0, 0, 0, 0, 97, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 128, 0, 0, 0, 128, 3, 0, 0, 0, 111, 0, 8, 128, 0, 0, 0, 2, 0, 0, 0, 0, 112, 0, 42, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 92, 6, 107, 71, 60, 12, 9, 7, 1, 7, 99, 3, 113, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs65 = { 5, { 87,88,89,-1,-1 } };
const TfArray<1, int> outputs65 = { 1, { 90 } };
const TfArray<2, int> inputs66 = { 2, { 90,31 } };
const TfArray<1, int> outputs66 = { 1, { 91 } };
uint8_t ALIGN(4) opdata67[50] = { 97, 0, 115, 0, 2, 0, 0, 64, 0, 2, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 224, 240, 1, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs67 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs67 = { 2, { 92,93 } };
const TfArray<5, int> inputs68 = { 5, { 91,92,93,-1,-1 } };
const TfArray<1, int> outputs68 = { 1, { 94 } };
const TfArray<2, int> inputs69 = { 2, { 94,31 } };
const TfArray<1, int> outputs69 = { 1, { 95 } };
const TfArray<1, int> inputs70 = { 1, { 95 } };
const TfArray<1, int> outputs70 = { 1, { 96 } };
uint8_t ALIGN(4) opdata71[50] = { 97, 0, 115, 0, 2, 0, 144, 4, 0, 2, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 64, 234, 1, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs71 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs71 = { 2, { 97,98 } };
const TfArray<5, int> inputs72 = { 5, { 96,97,98,-1,-1 } };
const TfArray<1, int> outputs72 = { 1, { 99 } };
const TfArray<2, int> inputs73 = { 2, { 99,31 } };
const TfArray<1, int> outputs73 = { 1, { 100 } };
uint8_t ALIGN(4) opdata74[50] = { 97, 0, 115, 0, 2, 0, 0, 64, 0, 2, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 64, 168, 1, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs74 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs74 = { 2, { 101,102 } };
const TfArray<5, int> inputs75 = { 5, { 100,101,102,-1,-1 } };
const TfArray<1, int> outputs75 = { 1, { 103 } };
const TfArray<2, int> inputs76 = { 2, { 103,31 } };
const TfArray<1, int> outputs76 = { 1, { 104 } };
uint8_t ALIGN(4) opdata77[52] = { 115, 0, 112, 0, 108, 0, 110, 0, 122, 0, 101, 0, 118, 0, 7, 5, 12, 11, 16, 19, 8, 13, 7, 0, 1, 0, 7, 0, 0, 5, 0, 4, 7, 0, 128, 0, 0, 0, 1, 0, 0, 0, 5, 5, 5, 5, 5, 105, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs77 = { 1, { 104 } };
const TfArray<1, int> outputs77 = { 1, { 105 } };
uint8_t ALIGN(4) opdata78[50] = { 97, 0, 115, 0, 2, 0, 144, 4, 0, 2, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 160, 161, 1, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs78 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs78 = { 2, { 106,107 } };
uint8_t ALIGN(4) opdata79[132] = { 109, 112, 0, 8, 0, 9, 0, 0, 0, 1, 0, 0, 0, 97, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 128, 0, 0, 0, 0, 3, 0, 0, 0, 111, 0, 8, 128, 0, 0, 0, 2, 0, 1, 0, 0, 112, 0, 42, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 92, 6, 107, 71, 60, 12, 9, 7, 1, 7, 99, 3, 113, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs79 = { 5, { 105,106,107,-1,-1 } };
const TfArray<1, int> outputs79 = { 1, { 108 } };
const TfArray<2, int> inputs80 = { 2, { 108,31 } };
const TfArray<1, int> outputs80 = { 1, { 109 } };
uint8_t ALIGN(4) opdata81[54] = { 97, 0, 115, 0, 2, 0, 0, 0, 0, 128, 0, 0, 0, 4, 0, 0, 6, 6, 116, 0, 3, 21, 20, 5, 3, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 160, 29, 1, 0, 32, 0, 0, 0, 0, 0, 0, 0, 6, 42, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs81 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs81 = { 2, { 110,111 } };
uint8_t ALIGN(4) opdata82[136] = { 109, 112, 0, 8, 0, 2, 0, 0, 128, 0, 0, 0, 0, 97, 0, 24, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 128, 1, 0, 0, 0, 111, 0, 8, 0, 1, 0, 0, 3, 0, 255, 255, 0, 112, 0, 42, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 96, 6, 111, 71, 60, 12, 9, 7, 1, 7, 103, 0, 117, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs82 = { 5, { 109,110,111,-1,-1 } };
const TfArray<1, int> outputs82 = { 1, { 112 } };
const TfArray<2, int> inputs83 = { 2, { 112,31 } };
const TfArray<1, int> outputs83 = { 1, { 113 } };
uint8_t ALIGN(4) opdata84[52] = { 115, 0, 112, 0, 108, 0, 110, 0, 122, 0, 101, 0, 118, 0, 7, 5, 12, 11, 16, 19, 8, 13, 7, 0, 1, 0, 7, 0, 0, 7, 0, 4, 3, 0, 0, 2, 0, 7, 1, 0, 0, 0, 5, 5, 5, 5, 5, 105, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs84 = { 1, { 113 } };
const TfArray<1, int> outputs84 = { 1, { 114 } };
uint8_t ALIGN(4) opdata85[50] = { 97, 0, 115, 0, 2, 0, 16, 9, 0, 4, 5, 5, 116, 0, 3, 15, 14, 5, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 128, 16, 1, 0, 30, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs85 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs85 = { 2, { 115,116 } };
uint8_t ALIGN(4) opdata86[132] = { 109, 112, 0, 8, 0, 6, 0, 0, 0, 1, 0, 0, 0, 97, 0, 20, 144, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 111, 0, 8, 0, 1, 0, 0, 2, 0, 1, 0, 0, 112, 0, 42, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 92, 6, 107, 71, 60, 12, 9, 7, 1, 7, 99, 3, 113, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs86 = { 5, { 114,115,116,-1,-1 } };
const TfArray<1, int> outputs86 = { 1, { 117 } };
const TfArray<2, int> inputs87 = { 2, { 117,31 } };
const TfArray<1, int> outputs87 = { 1, { 118 } };
uint8_t ALIGN(4) opdata88[42] = { 97, 0, 115, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 4, 0, 0, 6, 6, 116, 0, 3, 21, 20, 5, 3, 0, 1, 0, 3, 0, 128, 12, 24, 0, 0, 0, 5, 42, 5, 9, 37, 1,  }; /* custom_initial_data */
const int inputs88 = 0; /* empty TfLiteIntArray */
const TfArray<2, int> outputs88 = { 2, { 119,120 } };
uint8_t ALIGN(4) opdata89[136] = { 109, 112, 0, 8, 0, 4, 0, 0, 0, 1, 0, 0, 0, 97, 0, 24, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 111, 0, 8, 0, 1, 0, 0, 3, 0, 255, 255, 0, 112, 0, 42, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 96, 6, 111, 71, 60, 12, 9, 7, 1, 7, 103, 0, 117, 78, 23, 0, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs89 = { 5, { 118,119,120,-1,-1 } };
const TfArray<1, int> outputs89 = { 1, { 121 } };
const TfArray<2, int> inputs90 = { 2, { 121,31 } };
const TfArray<1, int> outputs90 = { 1, { 122 } };
uint8_t ALIGN(4) opdata91[63] = { 115, 0, 109, 0, 101, 0, 105, 0, 0, 0, 0, 0, 111, 0, 0, 0, 0, 0, 0, 0, 115, 109, 0, 0, 0, 0, 128, 61, 6, 25, 24, 29, 20, 33, 14, 0, 7, 0, 1, 0, 6, 0, 0, 1, 36, 0, 16, 0, 32, 0, 1, 0, 28, 0, 5, 34, 5, 34, 5, 34, 18, 37, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs91 = { 1, { 122 } };
const TfArray<1, int> outputs91 = { 1, { 123 } };
uint8_t ALIGN(4) opdata92[27] = { 97, 0, 115, 0, 1, 0, 128, 12, 5, 116, 0, 3, 12, 11, 5, 3, 1, 3, 0, 13, 0, 4, 41, 4, 6, 36, 1,  }; /* custom_initial_data */
const int inputs92 = 0; /* empty TfLiteIntArray */
const TfArray<1, int> outputs92 = { 1, { 125 } };
uint8_t ALIGN(4) opdata93[162] = { 109, 112, 0, 40, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 224, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0, 97, 0, 8, 12, 0, 0, 0, 0, 1, 0, 0, 0, 111, 0, 8, 12, 0, 0, 0, 2, 0, 255, 255, 0, 112, 0, 42, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 80, 6, 127, 71, 60, 12, 9, 7, 0, 1, 0, 7, 0, 90, 0, 1, 0, 138, 0, 84, 0, 30, 0, 32, 1, 0, 0, 20, 5, 20, 20, 40, 5, 5, 21, 37, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs93 = { 5, { 123,125,1,-1,124 } };
const TfArray<1, int> outputs93 = { 1, { 126 } };
uint8_t ALIGN(4) opdata94[32] = { 115, 0, 111, 0, 108, 0, 110, 0, 118, 0, 5, 7, 6, 11, 14, 7, 5, 1, 5, 10, 1, 12, 0, 0, 4, 4, 4, 4, 104, 10, 36, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs94 = { 1, { 126 } };
const TfArray<1, int> outputs94 = { 1, { 127 } };
const TfArray<1, int> inputs95 = { 1, { 127 } };
const TfArray<1, int> outputs95 = { 1, { 128 } };
uint8_t ALIGN(4) opdata96[46] = { 97, 0, 115, 0, 1, 0, 0, 4, 5, 116, 0, 3, 12, 11, 5, 0, 4, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 224, 103, 3, 0, 26, 0, 0, 0, 0, 0, 0, 0, 6, 41, 6, 15, 38, 1,  }; /* custom_initial_data */
const int inputs96 = 0; /* empty TfLiteIntArray */
const TfArray<1, int> outputs96 = { 1, { 129 } };
const TfArray<2, int> inputs97 = { 2, { 128,129 } };
const TfArray<1, int> outputs97 = { 1, { 130 } };
} g0;

TfLiteTensor tflTensors[] = 
{{ {(int32_t*)(tensor_arena + 17416)},(TfLiteIntArray*)&g0.tensor_dimension0, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant0)) }, {g0.quant0.scale->data[0], g0.quant0.zero_point->data[0] },},
{ {(int32_t*)g0.tensor_data1},(TfLiteIntArray*)&g0.tensor_dimension1, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)g0.tensor_data2},(TfLiteIntArray*)&g0.tensor_dimension2, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)g0.tensor_data3},(TfLiteIntArray*)&g0.tensor_dimension2, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)g0.tensor_data4},(TfLiteIntArray*)&g0.tensor_dimension4, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)g0.tensor_data5},(TfLiteIntArray*)&g0.tensor_dimension4, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 1032)},(TfLiteIntArray*)&g0.tensor_dimension6, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant0)) }, {g0.quant0.scale->data[0], g0.quant0.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension7, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant0)) }, {g0.quant0.scale->data[0], g0.quant0.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 102184)},(TfLiteIntArray*)&g0.tensor_dimension8, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 101416)},(TfLiteIntArray*)&g0.tensor_dimension9, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 68648)},(TfLiteIntArray*)&g0.tensor_dimension10, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant10)) }, {g0.quant10.scale->data[0], g0.quant10.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 66568)},(TfLiteIntArray*)&g0.tensor_dimension11, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant10)) }, {g0.quant10.scale->data[0], g0.quant10.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension12, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 101416)},(TfLiteIntArray*)&g0.tensor_dimension10, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant13)) }, {g0.quant13.scale->data[0], g0.quant13.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 68112)},(TfLiteIntArray*)&g0.tensor_dimension14, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 67600)},(TfLiteIntArray*)&g0.tensor_dimension15, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 2064)},(TfLiteIntArray*)&g0.tensor_dimension16, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant16)) }, {g0.quant16.scale->data[0], g0.quant16.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension17, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant16)) }, {g0.quant16.scale->data[0], g0.quant16.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 83984)},(TfLiteIntArray*)&g0.tensor_dimension12, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 67600)},(TfLiteIntArray*)&g0.tensor_dimension19, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant19)) }, {g0.quant19.scale->data[0], g0.quant19.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 37888)},(TfLiteIntArray*)&g0.tensor_dimension14, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 36992)},(TfLiteIntArray*)&g0.tensor_dimension9, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 37760)},(TfLiteIntArray*)&g0.tensor_dimension14, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 4224)},(TfLiteIntArray*)&g0.tensor_dimension23, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant23)) }, {g0.quant23.scale->data[0], g0.quant23.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension24, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant23)) }, {g0.quant23.scale->data[0], g0.quant23.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 36992)},(TfLiteIntArray*)&g0.tensor_dimension25, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 37296)},(TfLiteIntArray*)&g0.tensor_dimension14, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 67616)},(TfLiteIntArray*)&g0.tensor_dimension23, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant27)) }, {g0.quant27.scale->data[0], g0.quant27.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension28, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 1024)},(TfLiteIntArray*)&g0.tensor_dimension14, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 34848)},(TfLiteIntArray*)&g0.tensor_dimension23, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 74752)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteUInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 2080)},(TfLiteIntArray*)&g0.tensor_dimension23, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension33, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 43040)},(TfLiteIntArray*)&g0.tensor_dimension25, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 43344)},(TfLiteIntArray*)&g0.tensor_dimension14, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 34848)},(TfLiteIntArray*)&g0.tensor_dimension36, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension36, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 8192)},(TfLiteIntArray*)&g0.tensor_dimension38, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 10240)},(TfLiteIntArray*)&g0.tensor_dimension39, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 20736)},(TfLiteIntArray*)&g0.tensor_dimension40, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 4352)},(TfLiteIntArray*)&g0.tensor_dimension40, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension42, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 37120)},(TfLiteIntArray*)&g0.tensor_dimension43, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 37712)},(TfLiteIntArray*)&g0.tensor_dimension39, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 20736)},(TfLiteIntArray*)&g0.tensor_dimension40, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension40, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 34880)},(TfLiteIntArray*)&g0.tensor_dimension47, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 16384)},(TfLiteIntArray*)&g0.tensor_dimension39, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 18496)},(TfLiteIntArray*)&g0.tensor_dimension40, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 2112)},(TfLiteIntArray*)&g0.tensor_dimension40, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension51, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 22592)},(TfLiteIntArray*)&g0.tensor_dimension43, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 23184)},(TfLiteIntArray*)&g0.tensor_dimension39, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 18496)},(TfLiteIntArray*)&g0.tensor_dimension54, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 8192)},(TfLiteIntArray*)&g0.tensor_dimension54, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension56, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 12288)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 12800)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 4608)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension60, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 20992)},(TfLiteIntArray*)&g0.tensor_dimension61, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 22160)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 12800)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 24576)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension65, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 16384)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 4608)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension60, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 20992)},(TfLiteIntArray*)&g0.tensor_dimension61, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 22160)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 12800)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 24576)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension65, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 16384)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 4608)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension60, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 20992)},(TfLiteIntArray*)&g0.tensor_dimension61, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 22160)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 12800)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 24576)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension65, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 16384)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 4608)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension60, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 20992)},(TfLiteIntArray*)&g0.tensor_dimension61, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 22160)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 12800)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 24576)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension65, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 16384)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 4608)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension60, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 20992)},(TfLiteIntArray*)&g0.tensor_dimension61, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 22160)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 12800)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 24576)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension65, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 16384)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 2176)},(TfLiteIntArray*)&g0.tensor_dimension58, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension105, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 12416)},(TfLiteIntArray*)&g0.tensor_dimension61, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 13584)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 10368)},(TfLiteIntArray*)&g0.tensor_dimension108, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 36864)},(TfLiteIntArray*)&g0.tensor_dimension108, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension110, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 38912)},(TfLiteIntArray*)&g0.tensor_dimension15, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 32768)},(TfLiteIntArray*)&g0.tensor_dimension112, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 5120)},(TfLiteIntArray*)&g0.tensor_dimension112, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension114, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 13312)},(TfLiteIntArray*)&g0.tensor_dimension115, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 15632)},(TfLiteIntArray*)&g0.tensor_dimension15, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 9216)},(TfLiteIntArray*)&g0.tensor_dimension112, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 69632)},(TfLiteIntArray*)&g0.tensor_dimension112, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension119, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 73728)},(TfLiteIntArray*)&g0.tensor_dimension15, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 65536)},(TfLiteIntArray*)&g0.tensor_dimension112, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension112, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 4096)},(TfLiteIntArray*)&g0.tensor_dimension123, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 3200)},(TfLiteIntArray*)&g0.tensor_dimension124, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension125, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 3488)},(TfLiteIntArray*)&g0.tensor_dimension126, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 1040)},(TfLiteIntArray*)&g0.tensor_dimension127, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 1040)},(TfLiteIntArray*)&g0.tensor_dimension128, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant30)) }, {g0.quant30.scale->data[0], g0.quant30.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension31, kTfLiteFloat32, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 1024)},(TfLiteIntArray*)&g0.tensor_dimension128, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant130)) }, {g0.quant130.scale->data[0], g0.quant130.zero_point->data[0] },},
};

TfLiteNode tflNodes[] = 
{{ (TfLiteIntArray*)&g0.inputs0, (TfLiteIntArray*)&g0.outputs0, (TfLiteIntArray*)&g0.inputs0, const_cast<void*>(static_cast<const void*>(&g0.opdata0)), 28, },
{ (TfLiteIntArray*)&g0.inputs1, (TfLiteIntArray*)&g0.outputs1, (TfLiteIntArray*)&g0.inputs1, const_cast<void*>(static_cast<const void*>(&g0.opdata1)), 74, },
{ (TfLiteIntArray*)&g0.inputs2, (TfLiteIntArray*)&g0.outputs2, (TfLiteIntArray*)&g0.inputs2, const_cast<void*>(static_cast<const void*>(&g0.opdata2)), 46, },
{ (TfLiteIntArray*)&g0.inputs3, (TfLiteIntArray*)&g0.outputs3, (TfLiteIntArray*)&g0.inputs3, const_cast<void*>(static_cast<const void*>(&g0.opdata3)), 152, },
{ (TfLiteIntArray*)&g0.inputs4, (TfLiteIntArray*)&g0.outputs4, (TfLiteIntArray*)&g0.inputs4, const_cast<void*>(static_cast<const void*>(&g0.opdata4)), 74, },
{ (TfLiteIntArray*)&g0.inputs5, (TfLiteIntArray*)&g0.outputs5, (TfLiteIntArray*)&g0.inputs5, const_cast<void*>(static_cast<const void*>(&g0.opdata5)), 46, },
{ (TfLiteIntArray*)&g0.inputs6, (TfLiteIntArray*)&g0.outputs6, (TfLiteIntArray*)&g0.inputs6, const_cast<void*>(static_cast<const void*>(&g0.opdata6)), 132, },
{ (TfLiteIntArray*)&g0.inputs7, (TfLiteIntArray*)&g0.outputs7, (TfLiteIntArray*)&g0.inputs7, const_cast<void*>(static_cast<const void*>(&g0.opdata7)), 46, },
{ (TfLiteIntArray*)&g0.inputs8, (TfLiteIntArray*)&g0.outputs8, (TfLiteIntArray*)&g0.inputs8, const_cast<void*>(static_cast<const void*>(&g0.opdata8)), 152, },
{ (TfLiteIntArray*)&g0.inputs9, (TfLiteIntArray*)&g0.outputs9, (TfLiteIntArray*)&g0.inputs9, const_cast<void*>(static_cast<const void*>(&g0.opdata9)), 74, },
{ (TfLiteIntArray*)&g0.inputs10, (TfLiteIntArray*)&g0.outputs10, (TfLiteIntArray*)&g0.inputs10, const_cast<void*>(static_cast<const void*>(&g0.opdata10)), 46, },
{ (TfLiteIntArray*)&g0.inputs11, (TfLiteIntArray*)&g0.outputs11, (TfLiteIntArray*)&g0.inputs11, const_cast<void*>(static_cast<const void*>(&g0.opdata11)), 132, },
{ (TfLiteIntArray*)&g0.inputs12, (TfLiteIntArray*)&g0.outputs12, (TfLiteIntArray*)&g0.inputs12, const_cast<void*>(static_cast<const void*>(&g0.opdata12)), 50, },
{ (TfLiteIntArray*)&g0.inputs13, (TfLiteIntArray*)&g0.outputs13, (TfLiteIntArray*)&g0.inputs13, const_cast<void*>(static_cast<const void*>(&g0.opdata13)), 152, },
{ (TfLiteIntArray*)&g0.inputs14, (TfLiteIntArray*)&g0.outputs14, (TfLiteIntArray*)&g0.inputs14, const_cast<void*>(static_cast<const void*>(&g0.opdata14)), 74, },
{ (TfLiteIntArray*)&g0.inputs15, (TfLiteIntArray*)&g0.outputs15, (TfLiteIntArray*)&g0.inputs15, const_cast<void*>(static_cast<const void*>(&g0.opdata15)), 50, },
{ (TfLiteIntArray*)&g0.inputs16, (TfLiteIntArray*)&g0.outputs16, (TfLiteIntArray*)&g0.inputs16, const_cast<void*>(static_cast<const void*>(&g0.opdata16)), 132, },
{ (TfLiteIntArray*)&g0.inputs17, (TfLiteIntArray*)&g0.outputs17, (TfLiteIntArray*)&g0.inputs17, const_cast<void*>(static_cast<const void*>(&g0.opdata17)), 50, },
{ (TfLiteIntArray*)&g0.inputs18, (TfLiteIntArray*)&g0.outputs18, (TfLiteIntArray*)&g0.inputs18, const_cast<void*>(static_cast<const void*>(&g0.opdata18)), 136, },
{ (TfLiteIntArray*)&g0.inputs19, (TfLiteIntArray*)&g0.outputs19, (TfLiteIntArray*)&g0.inputs19, const_cast<void*>(static_cast<const void*>(&g0.opdata19)), 46, },
{ (TfLiteIntArray*)&g0.inputs20, (TfLiteIntArray*)&g0.outputs20, (TfLiteIntArray*)&g0.inputs20, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs21, (TfLiteIntArray*)&g0.outputs21, (TfLiteIntArray*)&g0.inputs21, const_cast<void*>(static_cast<const void*>(&g0.opdata21)), 52, },
{ (TfLiteIntArray*)&g0.inputs22, (TfLiteIntArray*)&g0.outputs22, (TfLiteIntArray*)&g0.inputs22, const_cast<void*>(static_cast<const void*>(&g0.opdata22)), 50, },
{ (TfLiteIntArray*)&g0.inputs23, (TfLiteIntArray*)&g0.outputs23, (TfLiteIntArray*)&g0.inputs23, const_cast<void*>(static_cast<const void*>(&g0.opdata23)), 132, },
{ (TfLiteIntArray*)&g0.inputs24, (TfLiteIntArray*)&g0.outputs24, (TfLiteIntArray*)&g0.inputs24, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs25, (TfLiteIntArray*)&g0.outputs25, (TfLiteIntArray*)&g0.inputs25, const_cast<void*>(static_cast<const void*>(&g0.opdata25)), 50, },
{ (TfLiteIntArray*)&g0.inputs26, (TfLiteIntArray*)&g0.outputs26, (TfLiteIntArray*)&g0.inputs26, const_cast<void*>(static_cast<const void*>(&g0.opdata26)), 136, },
{ (TfLiteIntArray*)&g0.inputs27, (TfLiteIntArray*)&g0.outputs27, (TfLiteIntArray*)&g0.inputs27, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs28, (TfLiteIntArray*)&g0.outputs28, (TfLiteIntArray*)&g0.inputs28, const_cast<void*>(static_cast<const void*>(&g0.opdata28)), 52, },
{ (TfLiteIntArray*)&g0.inputs29, (TfLiteIntArray*)&g0.outputs29, (TfLiteIntArray*)&g0.inputs29, const_cast<void*>(static_cast<const void*>(&g0.opdata29)), 50, },
{ (TfLiteIntArray*)&g0.inputs30, (TfLiteIntArray*)&g0.outputs30, (TfLiteIntArray*)&g0.inputs30, const_cast<void*>(static_cast<const void*>(&g0.opdata30)), 132, },
{ (TfLiteIntArray*)&g0.inputs31, (TfLiteIntArray*)&g0.outputs31, (TfLiteIntArray*)&g0.inputs31, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs32, (TfLiteIntArray*)&g0.outputs32, (TfLiteIntArray*)&g0.inputs32, const_cast<void*>(static_cast<const void*>(&g0.opdata32)), 50, },
{ (TfLiteIntArray*)&g0.inputs33, (TfLiteIntArray*)&g0.outputs33, (TfLiteIntArray*)&g0.inputs33, const_cast<void*>(static_cast<const void*>(&g0.opdata33)), 136, },
{ (TfLiteIntArray*)&g0.inputs34, (TfLiteIntArray*)&g0.outputs34, (TfLiteIntArray*)&g0.inputs34, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs35, (TfLiteIntArray*)&g0.outputs35, (TfLiteIntArray*)&g0.inputs35, const_cast<void*>(static_cast<const void*>(&g0.opdata35)), 52, },
{ (TfLiteIntArray*)&g0.inputs36, (TfLiteIntArray*)&g0.outputs36, (TfLiteIntArray*)&g0.inputs36, const_cast<void*>(static_cast<const void*>(&g0.opdata36)), 50, },
{ (TfLiteIntArray*)&g0.inputs37, (TfLiteIntArray*)&g0.outputs37, (TfLiteIntArray*)&g0.inputs37, const_cast<void*>(static_cast<const void*>(&g0.opdata37)), 132, },
{ (TfLiteIntArray*)&g0.inputs38, (TfLiteIntArray*)&g0.outputs38, (TfLiteIntArray*)&g0.inputs38, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs39, (TfLiteIntArray*)&g0.outputs39, (TfLiteIntArray*)&g0.inputs39, const_cast<void*>(static_cast<const void*>(&g0.opdata39)), 50, },
{ (TfLiteIntArray*)&g0.inputs40, (TfLiteIntArray*)&g0.outputs40, (TfLiteIntArray*)&g0.inputs40, const_cast<void*>(static_cast<const void*>(&g0.opdata40)), 136, },
{ (TfLiteIntArray*)&g0.inputs41, (TfLiteIntArray*)&g0.outputs41, (TfLiteIntArray*)&g0.inputs41, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs42, (TfLiteIntArray*)&g0.outputs42, (TfLiteIntArray*)&g0.inputs42, const_cast<void*>(static_cast<const void*>(&g0.opdata42)), 52, },
{ (TfLiteIntArray*)&g0.inputs43, (TfLiteIntArray*)&g0.outputs43, (TfLiteIntArray*)&g0.inputs43, const_cast<void*>(static_cast<const void*>(&g0.opdata43)), 50, },
{ (TfLiteIntArray*)&g0.inputs44, (TfLiteIntArray*)&g0.outputs44, (TfLiteIntArray*)&g0.inputs44, const_cast<void*>(static_cast<const void*>(&g0.opdata44)), 132, },
{ (TfLiteIntArray*)&g0.inputs45, (TfLiteIntArray*)&g0.outputs45, (TfLiteIntArray*)&g0.inputs45, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs46, (TfLiteIntArray*)&g0.outputs46, (TfLiteIntArray*)&g0.inputs46, const_cast<void*>(static_cast<const void*>(&g0.opdata46)), 50, },
{ (TfLiteIntArray*)&g0.inputs47, (TfLiteIntArray*)&g0.outputs47, (TfLiteIntArray*)&g0.inputs47, const_cast<void*>(static_cast<const void*>(&g0.opdata47)), 136, },
{ (TfLiteIntArray*)&g0.inputs48, (TfLiteIntArray*)&g0.outputs48, (TfLiteIntArray*)&g0.inputs48, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs49, (TfLiteIntArray*)&g0.outputs49, (TfLiteIntArray*)&g0.inputs49, const_cast<void*>(static_cast<const void*>(&g0.opdata42)), 52, },
{ (TfLiteIntArray*)&g0.inputs50, (TfLiteIntArray*)&g0.outputs50, (TfLiteIntArray*)&g0.inputs50, const_cast<void*>(static_cast<const void*>(&g0.opdata50)), 50, },
{ (TfLiteIntArray*)&g0.inputs51, (TfLiteIntArray*)&g0.outputs51, (TfLiteIntArray*)&g0.inputs51, const_cast<void*>(static_cast<const void*>(&g0.opdata44)), 132, },
{ (TfLiteIntArray*)&g0.inputs52, (TfLiteIntArray*)&g0.outputs52, (TfLiteIntArray*)&g0.inputs52, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs53, (TfLiteIntArray*)&g0.outputs53, (TfLiteIntArray*)&g0.inputs53, const_cast<void*>(static_cast<const void*>(&g0.opdata53)), 50, },
{ (TfLiteIntArray*)&g0.inputs54, (TfLiteIntArray*)&g0.outputs54, (TfLiteIntArray*)&g0.inputs54, const_cast<void*>(static_cast<const void*>(&g0.opdata47)), 136, },
{ (TfLiteIntArray*)&g0.inputs55, (TfLiteIntArray*)&g0.outputs55, (TfLiteIntArray*)&g0.inputs55, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs56, (TfLiteIntArray*)&g0.outputs56, (TfLiteIntArray*)&g0.inputs56, const_cast<void*>(static_cast<const void*>(&g0.opdata42)), 52, },
{ (TfLiteIntArray*)&g0.inputs57, (TfLiteIntArray*)&g0.outputs57, (TfLiteIntArray*)&g0.inputs57, const_cast<void*>(static_cast<const void*>(&g0.opdata57)), 50, },
{ (TfLiteIntArray*)&g0.inputs58, (TfLiteIntArray*)&g0.outputs58, (TfLiteIntArray*)&g0.inputs58, const_cast<void*>(static_cast<const void*>(&g0.opdata44)), 132, },
{ (TfLiteIntArray*)&g0.inputs59, (TfLiteIntArray*)&g0.outputs59, (TfLiteIntArray*)&g0.inputs59, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs60, (TfLiteIntArray*)&g0.outputs60, (TfLiteIntArray*)&g0.inputs60, const_cast<void*>(static_cast<const void*>(&g0.opdata60)), 50, },
{ (TfLiteIntArray*)&g0.inputs61, (TfLiteIntArray*)&g0.outputs61, (TfLiteIntArray*)&g0.inputs61, const_cast<void*>(static_cast<const void*>(&g0.opdata47)), 136, },
{ (TfLiteIntArray*)&g0.inputs62, (TfLiteIntArray*)&g0.outputs62, (TfLiteIntArray*)&g0.inputs62, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs63, (TfLiteIntArray*)&g0.outputs63, (TfLiteIntArray*)&g0.inputs63, const_cast<void*>(static_cast<const void*>(&g0.opdata42)), 52, },
{ (TfLiteIntArray*)&g0.inputs64, (TfLiteIntArray*)&g0.outputs64, (TfLiteIntArray*)&g0.inputs64, const_cast<void*>(static_cast<const void*>(&g0.opdata64)), 50, },
{ (TfLiteIntArray*)&g0.inputs65, (TfLiteIntArray*)&g0.outputs65, (TfLiteIntArray*)&g0.inputs65, const_cast<void*>(static_cast<const void*>(&g0.opdata65)), 132, },
{ (TfLiteIntArray*)&g0.inputs66, (TfLiteIntArray*)&g0.outputs66, (TfLiteIntArray*)&g0.inputs66, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs67, (TfLiteIntArray*)&g0.outputs67, (TfLiteIntArray*)&g0.inputs67, const_cast<void*>(static_cast<const void*>(&g0.opdata67)), 50, },
{ (TfLiteIntArray*)&g0.inputs68, (TfLiteIntArray*)&g0.outputs68, (TfLiteIntArray*)&g0.inputs68, const_cast<void*>(static_cast<const void*>(&g0.opdata47)), 136, },
{ (TfLiteIntArray*)&g0.inputs69, (TfLiteIntArray*)&g0.outputs69, (TfLiteIntArray*)&g0.inputs69, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs70, (TfLiteIntArray*)&g0.outputs70, (TfLiteIntArray*)&g0.inputs70, const_cast<void*>(static_cast<const void*>(&g0.opdata42)), 52, },
{ (TfLiteIntArray*)&g0.inputs71, (TfLiteIntArray*)&g0.outputs71, (TfLiteIntArray*)&g0.inputs71, const_cast<void*>(static_cast<const void*>(&g0.opdata71)), 50, },
{ (TfLiteIntArray*)&g0.inputs72, (TfLiteIntArray*)&g0.outputs72, (TfLiteIntArray*)&g0.inputs72, const_cast<void*>(static_cast<const void*>(&g0.opdata65)), 132, },
{ (TfLiteIntArray*)&g0.inputs73, (TfLiteIntArray*)&g0.outputs73, (TfLiteIntArray*)&g0.inputs73, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs74, (TfLiteIntArray*)&g0.outputs74, (TfLiteIntArray*)&g0.inputs74, const_cast<void*>(static_cast<const void*>(&g0.opdata74)), 50, },
{ (TfLiteIntArray*)&g0.inputs75, (TfLiteIntArray*)&g0.outputs75, (TfLiteIntArray*)&g0.inputs75, const_cast<void*>(static_cast<const void*>(&g0.opdata47)), 136, },
{ (TfLiteIntArray*)&g0.inputs76, (TfLiteIntArray*)&g0.outputs76, (TfLiteIntArray*)&g0.inputs76, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs77, (TfLiteIntArray*)&g0.outputs77, (TfLiteIntArray*)&g0.inputs77, const_cast<void*>(static_cast<const void*>(&g0.opdata77)), 52, },
{ (TfLiteIntArray*)&g0.inputs78, (TfLiteIntArray*)&g0.outputs78, (TfLiteIntArray*)&g0.inputs78, const_cast<void*>(static_cast<const void*>(&g0.opdata78)), 50, },
{ (TfLiteIntArray*)&g0.inputs79, (TfLiteIntArray*)&g0.outputs79, (TfLiteIntArray*)&g0.inputs79, const_cast<void*>(static_cast<const void*>(&g0.opdata79)), 132, },
{ (TfLiteIntArray*)&g0.inputs80, (TfLiteIntArray*)&g0.outputs80, (TfLiteIntArray*)&g0.inputs80, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs81, (TfLiteIntArray*)&g0.outputs81, (TfLiteIntArray*)&g0.inputs81, const_cast<void*>(static_cast<const void*>(&g0.opdata81)), 54, },
{ (TfLiteIntArray*)&g0.inputs82, (TfLiteIntArray*)&g0.outputs82, (TfLiteIntArray*)&g0.inputs82, const_cast<void*>(static_cast<const void*>(&g0.opdata82)), 136, },
{ (TfLiteIntArray*)&g0.inputs83, (TfLiteIntArray*)&g0.outputs83, (TfLiteIntArray*)&g0.inputs83, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs84, (TfLiteIntArray*)&g0.outputs84, (TfLiteIntArray*)&g0.inputs84, const_cast<void*>(static_cast<const void*>(&g0.opdata84)), 52, },
{ (TfLiteIntArray*)&g0.inputs85, (TfLiteIntArray*)&g0.outputs85, (TfLiteIntArray*)&g0.inputs85, const_cast<void*>(static_cast<const void*>(&g0.opdata85)), 50, },
{ (TfLiteIntArray*)&g0.inputs86, (TfLiteIntArray*)&g0.outputs86, (TfLiteIntArray*)&g0.inputs86, const_cast<void*>(static_cast<const void*>(&g0.opdata86)), 132, },
{ (TfLiteIntArray*)&g0.inputs87, (TfLiteIntArray*)&g0.outputs87, (TfLiteIntArray*)&g0.inputs87, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs88, (TfLiteIntArray*)&g0.outputs88, (TfLiteIntArray*)&g0.inputs88, const_cast<void*>(static_cast<const void*>(&g0.opdata88)), 42, },
{ (TfLiteIntArray*)&g0.inputs89, (TfLiteIntArray*)&g0.outputs89, (TfLiteIntArray*)&g0.inputs89, const_cast<void*>(static_cast<const void*>(&g0.opdata89)), 136, },
{ (TfLiteIntArray*)&g0.inputs90, (TfLiteIntArray*)&g0.outputs90, (TfLiteIntArray*)&g0.inputs90, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs91, (TfLiteIntArray*)&g0.outputs91, (TfLiteIntArray*)&g0.inputs91, const_cast<void*>(static_cast<const void*>(&g0.opdata91)), 63, },
{ (TfLiteIntArray*)&g0.inputs92, (TfLiteIntArray*)&g0.outputs92, (TfLiteIntArray*)&g0.inputs92, const_cast<void*>(static_cast<const void*>(&g0.opdata92)), 27, },
{ (TfLiteIntArray*)&g0.inputs93, (TfLiteIntArray*)&g0.outputs93, (TfLiteIntArray*)&g0.inputs93, const_cast<void*>(static_cast<const void*>(&g0.opdata93)), 162, },
{ (TfLiteIntArray*)&g0.inputs94, (TfLiteIntArray*)&g0.outputs94, (TfLiteIntArray*)&g0.inputs94, const_cast<void*>(static_cast<const void*>(&g0.opdata94)), 32, },
{ (TfLiteIntArray*)&g0.inputs95, (TfLiteIntArray*)&g0.outputs95, (TfLiteIntArray*)&g0.inputs95, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
{ (TfLiteIntArray*)&g0.inputs96, (TfLiteIntArray*)&g0.outputs96, (TfLiteIntArray*)&g0.inputs96, const_cast<void*>(static_cast<const void*>(&g0.opdata96)), 46, },
{ (TfLiteIntArray*)&g0.inputs97, (TfLiteIntArray*)&g0.outputs97, (TfLiteIntArray*)&g0.inputs97, const_cast<void*>(static_cast<const void*>(&g0.opdata20)), 0, },
};

used_operators_e used_ops[] =
{OP_XC_pad_3_to_4, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_ld_weights, OP_XC_lookup, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_pad, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_lookup, OP_XC_mean, OP_XC_ld_weights, OP_XC_conv2d_v2, OP_XC_slice, OP_XC_no_op, OP_XC_ld_weights, OP_XC_softmax, };


// Indices into tflTensors and tflNodes for subgraphs
size_t tflTensors_subgraph_index[] = {0, 131, };
size_t tflNodes_subgraph_index[] = {0, 98, };

// Variable tensors
size_t varTensors_index[] = {};

// Input/output tensors
static const int inTensorIndices[] = {
  0, 
};

static const int outTensorIndices[] = {
  130, 
};

static const int externalInOutTensorIndices[] = {
  
};
static const int externalInOutTensorOffsets[] = {
  
};

// Indices into inTensors and outTensors for subgraphs
size_t inTensors_subgraph_index[] = {0, 1, };
size_t outTensors_subgraph_index[] = {0, 1, };

// Indices for output tensors that are modified by certain TFLite ops and
// have to be reset in model_init() if the tensor arena gets trashed
static const int tfliteModifiedOutputTensorIndices[] = {};
TfLiteIntArray* tfliteModifiedOutputTensorOriginalDims[] = {};

// Scratch buffer variables
int scratch_buffer_idx;
const int scratch_buffer_offsets[0] = {  };
tflite_micro::MicroContext mc;
tflite_micro::MicroGraph micro_graph;
size_t currentSubgraphIndex = 0;

// Xcore context and thread variables
xc_context_config_t xc_config;
// When using USE_DDR_FIX for enabling LPDDR support, only one thread can be used
#ifdef USE_DDR_FIX
static_assert((1 == 1),
             "Only one thread can be used when using USE_DDR_FIX! Please recompile with one thread!");
#endif

// Persistent buffer ptr
// Initialized to the tail end of the tensor arena
uint8_t *persistentBufferPtr;
// Functions to be used as function pointers for TfLiteContext and MicroContext 
static void* AllocatePersistentBuffer(struct TfLiteContext* ctx,
                                                 size_t bytes) {
  // Align to double word
  bytes = ((bytes + 7) / 8) * 8;
  persistentBufferPtr -= bytes;
  return persistentBufferPtr;
}

static TfLiteEvalTensor *GetEvalTensor(const struct TfLiteContext *context,
                                       int tensor_idx) {
  return (TfLiteEvalTensor*)&tflTensors[tflTensors_subgraph_index[currentSubgraphIndex] + tensor_idx];
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

static bool IsConstantTensor(struct TfLiteContext *context,
                                       TfLiteTensor* tensor) {
  bool constant = true;
  if(tensor->data.data > &tensor_arena[0] && tensor->data.data < &tensor_arena[kTensorArenaSize - 1]){
    constant = false;
  }
  return constant;
}

static bool IsVariableTensor(struct TfLiteContext *context,
                                       TfLiteTensor* tensor) {
  bool found = false;
  for (int i = 0; i < 0; i++) {
    if(tensor == &tflTensors[varTensors_index[i]]){
      found = true;
    }
  }
  return found;
}

static size_t TensorBytes(TfLiteTensor *tensor) {
  int element_count = 1;
  for (int i = 0; i < tensor->dims->size; ++i) {
    element_count *= tensor->dims->data[i];
  }

  size_t bytes_per_element;
  tflite_micro::TfLiteTypeSizeOf(tensor->type, &bytes_per_element);
  return element_count * bytes_per_element;
}

static TfLiteTensor* mc_AllocateTempInputTensor(const TfLiteNode* node, int index) {
      if (node->inputs->data[index] < 0) {
        return nullptr;
      }
      return &ctx.tensors[tflTensors_subgraph_index[currentSubgraphIndex] + node->inputs->data[index]];
}

static TfLiteTensor* mc_AllocateTempOutputTensor(const TfLiteNode* node, int index) {
      if (node->outputs->data[index] < 0) {
        return nullptr;
      }
      return &ctx.tensors[tflTensors_subgraph_index[currentSubgraphIndex] + node->outputs->data[index]];
}

static void mc_DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
}

static void* mc_external_context() {
  return &xc_config;
}

static tflite_micro::MicroGraph& mc_graph() {
  return micro_graph;
}

static int mg_NumSubgraphs(){
  return sizeof(tflTensors_subgraph_index)/sizeof(size_t) - 1;
}

static size_t mg_NumSubgraphInputs(int subgraph_idx){
  return inTensors_subgraph_index[subgraph_idx+1] - inTensors_subgraph_index[subgraph_idx];
}

static size_t mg_NumSubgraphOutputs(int subgraph_idx){
  return outTensors_subgraph_index[subgraph_idx+1] - outTensors_subgraph_index[subgraph_idx];
}

static TfLiteEvalTensor* mg_GetSubgraphInput(int subgraph_idx, int i){
  return (TfLiteEvalTensor*)&tflTensors[tflTensors_subgraph_index[subgraph_idx] + inTensorIndices[inTensors_subgraph_index[subgraph_idx] + i]];
}

static TfLiteEvalTensor* mg_GetSubgraphOutput(int subgraph_idx, int i){
  return (TfLiteEvalTensor*)&tflTensors[tflTensors_subgraph_index[subgraph_idx] + outTensorIndices[outTensors_subgraph_index[subgraph_idx] + i]];
}

static TfLiteStatus mg_InvokeSubgraph(int g){
  int prevSubgraphIndex = currentSubgraphIndex;
  currentSubgraphIndex = g;
#ifdef TFLMC_PRINT_TENSORS
printf("[\n");
#endif

  for (size_t i = tflNodes_subgraph_index[g]; i < tflNodes_subgraph_index[g+1]; ++i) {

#ifdef TFLMC_PRINT_INPUT_TENSORS
    // print every input tensor
    printf("node in %d\n", i);
    for (int j = 0; j < tflNodes[i].inputs->size; j++) {
      // -1 such as in case of no bias tensor for conv
      if (tflNodes[i].inputs->data[j] != -1) {
        printf("tensor %d, input %d, %d bytes, checksum %d\n", tflNodes[i].inputs->data[j], j, TensorBytes(&tflTensors[tflNodes[i].inputs->data[j]]), checksum(tflTensors[tflNodes[i].inputs->data[j]].data.raw, TensorBytes(&tflTensors[tflNodes[i].inputs->data[j]])));
        for (int k = 0; k < TensorBytes(&tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].inputs->data[j]]); k++) {
          printf("%d,", (int8_t)tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].inputs->data[j]].data.raw[k]);
        }
        printf("\n");
      }
    }
    printf("\n");
#endif

#ifdef TFLMC_XCORE_PROFILE
#if defined(__xcore__) || defined(__riscv_xxcore)
    time_t0 = get_reference_time();
#endif
#endif

    TfLiteStatus status = registrations[used_ops[i]].invoke(&ctx, &tflNodes[i]);

    if (status != kTfLiteOk) {
#ifdef TFLMC_XCORE_PROFILE
      printf("\nERROR: Node %d (%s) invocation failed with status %d\n", i, op_strs[used_ops[i]], status);
      printf("Model invocation aborted\n\n");
#endif
      currentSubgraphIndex = prevSubgraphIndex;
      return status;
    }

#ifdef TFLMC_XCORE_PROFILE
#if defined(__xcore__) || defined(__riscv_xxcore)
    time_t1 = get_reference_time();
#endif
    op_times[used_ops[i]] += time_t1 - time_t0;
    op_counts[used_ops[i]] += 1;
    printf("node %-5d %-32s %-12d\n", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

#ifdef TFLMC_PRINT_TENSORS
    // print every output tensor
    printf("{\"node\" : \"%d\", \"op\" : \"%s\", \"data\" : [\n", i, op_strs[used_ops[i]]);
    for (int j = 0; j < tflNodes[i].outputs->size; j++) {
      printf("{\"tensor\" : %d, \"output\" : %d, \"offset\" : %d, \"bytes\" : %d, \"checksum\" : %d,\n", tflNodes[i].outputs->data[j], j, tflTensors[tflNodes[i].outputs->data[j]].data.uint8 - tensor_arena, TensorBytes(&tflTensors[tflNodes[i].outputs->data[j]]), checksum(tflTensors[tflNodes[i].outputs->data[j]].data.raw, TensorBytes(&tflTensors[tflNodes[i].outputs->data[j]])));
      printf("\"val\" : [");
      for (int k = 0; k < TensorBytes(&tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].outputs->data[j]]); k++) {
        printf("%d", (int8_t)tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].outputs->data[j]].data.raw[k]);
        if (k < TensorBytes(&tflTensors[tflTensors_subgraph_index[g] + tflNodes[i].outputs->data[j]]) - 1) {
          printf(",");
        }
      }
      if (j < tflNodes[i].outputs->size - 1) {
        printf("]},\n");
      } else {
        printf("]}]\n");
      }
    }

    if (i < ((tflNodes_subgraph_index[g+1] - tflNodes_subgraph_index[g]) - 1)) {
      printf("},\n");
    } else {
      printf("}\n");
    }
#endif
  }
#ifdef TFLMC_PRINT_TENSORS
printf("]\n");
#endif

  currentSubgraphIndex = prevSubgraphIndex;
  return kTfLiteOk;
}


void InitExternalTensors(void *paging_ptr){
  int len = sizeof(externalInOutTensorIndices) / sizeof(int);
  for (int n = 0; n < len; ++n) {
    auto t = &tflTensors[externalInOutTensorIndices[n]];
    t->data.data = (int32_t *)(((int8_t *)paging_ptr) + externalInOutTensorOffsets[n]);
  }
}

void ResetModifiedTFLiteOutputTensorDims(){
  int len = sizeof(tfliteModifiedOutputTensorIndices) / sizeof(int);
  for (int n = 0; n < len; ++n) {
    auto t = &tflTensors[tfliteModifiedOutputTensorIndices[n]];
    t->dims = tfliteModifiedOutputTensorOriginalDims[n];
  }
}

} // namespace

TfLiteTensor* model_input(int index) {
  return &ctx.tensors[inTensorIndices[index]];
}

TfLiteTensor* model_output(int index) {
  return &ctx.tensors[outTensorIndices[index]];
}

size_t model_input_size(int index) {
  return TensorBytes(model_input(index));
}

size_t model_output_size(int index) {
  return TensorBytes(model_output(index));
}


static __attribute__((always_inline))
TfLiteStatus model_init_with_paging(void *weights_data_ptr, void *paging_ptr) {

// Set target arch based on the compiled model
  SetNNTargetArch(nn_target_arch_t::TARGET_ARCH_XS3A);

  // Clear and initialize
  scratch_buffer_idx = 0;
  persistentBufferPtr = tensor_arena + kTensorArenaSize;

  // Set weights data ptr in xcore context config
  xc_config.weights_data_ptr = weights_data_ptr;
  // Set paging ptr in xcore context config
  xc_config.paging_ptr = paging_ptr;
  // Set thread count specified in the compiler
  xc_config.model_thread_count = 1;

  // Initialize externally allocated input/output tensors with paging_ptr
  InitExternalTensors(paging_ptr);

  // Reset output tensor dims that are modified by certain TFLite ops and
  // have to be reset if the tensor arena gets trashed
  ResetModifiedTFLiteOutputTensorDims();

  // Setup microcontext functions
  mc.AllocateTempInputTensor = &mc_AllocateTempInputTensor;
  mc.AllocateTempOutputTensor = &mc_AllocateTempOutputTensor;
  mc.DeallocateTempTfLiteTensor = &mc_DeallocateTempTfLiteTensor;
  mc.external_context = &mc_external_context;
  mc.graph = &mc_graph;

  micro_graph.NumSubgraphs = &mg_NumSubgraphs;
  micro_graph.NumSubgraphInputs = &mg_NumSubgraphInputs;
  micro_graph.NumSubgraphOutputs = &mg_NumSubgraphOutputs;
  micro_graph.GetSubgraphInput = &mg_GetSubgraphInput;
  micro_graph.GetSubgraphOutput = &mg_GetSubgraphOutput;
  micro_graph.InvokeSubgraph = &mg_InvokeSubgraph;

  // Setup tflitecontext functions
  ctx.AllocatePersistentBuffer = &AllocatePersistentBuffer;
  ctx.GetEvalTensor = &GetEvalTensor;
  ctx.RequestScratchBufferInArena = &RequestScratchBufferInArena;
  ctx.GetScratchBuffer = &GetScratchBuffer;
  ctx.IsConstantTensor = &IsConstantTensor;
  ctx.IsVariableTensor = &IsVariableTensor;

  // Set microcontext as the context ptr
  ctx.impl_ = (void*)&mc;
  ctx.tensors = tflTensors;
  ctx.tensors_size = 131;
  registrations[OP_XC_pad_3_to_4] = *(tflite_micro::ops::micro::xcore::Register_XC_pad_3_to_4());
  registrations[OP_XC_pad] = *(tflite_micro::ops::micro::xcore::Register_XC_pad());
  registrations[OP_XC_ld_weights] = *(tflite_micro::ops::micro::xcore::Register_XC_ld_weights());
  registrations[OP_XC_conv2d_v2] = *(tflite_micro::ops::micro::xcore::Register_XC_conv2d_v2());
  registrations[OP_XC_lookup] = *(tflite_micro::ops::micro::xcore::Register_XC_lookup());
  registrations[OP_XC_mean] = *(tflite_micro::ops::micro::xcore::Register_XC_mean());
  registrations[OP_XC_slice] = *(tflite_micro::ops::micro::xcore::Register_XC_slice());
  registrations[OP_XC_no_op] = *(tflite_micro::ops::micro::xcore::Register_XC_no_op());
  registrations[OP_XC_softmax] = *(tflite_micro::ops::micro::xcore::Register_XC_softmax());


  // Allocate persistent buffers for variable tensors
  for (int i = 0; i < 0; i++) {
    tflTensors[varTensors_index[i]].data.data = AllocatePersistentBuffer(&ctx, TensorBytes(&tflTensors[varTensors_index[i]]));
  }

#ifdef TFLMC_XCORE_PROFILE
  printf("Profiling init()...\n");
  memset(op_times, 0, sizeof(op_times));
  op_times_summed = 0;
#endif

  for(size_t g = 0; g < 1; ++g) {
    currentSubgraphIndex = g;
    for(size_t i = tflNodes_subgraph_index[g]; i < tflNodes_subgraph_index[g+1]; ++i) {
    if (registrations[used_ops[i]].init) {

#ifdef TFLMC_XCORE_PROFILE
#if defined(__xcore__) || defined(__riscv_xxcore)
      time_t0 = get_reference_time();
#endif
#endif

      tflNodes[i].user_data = registrations[used_ops[i]].init(&ctx, (const char*)tflNodes[i].builtin_data, tflNodes[i].custom_initial_data_size);

#ifdef TFLMC_XCORE_PROFILE
#if defined(__xcore__) || defined(__riscv_xxcore)
      time_t1 = get_reference_time();
#endif
      op_times[used_ops[i]] += time_t1 - time_t0;
      printf("node %-5d %-32s %-12d\n", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

    }
  }
  }
  currentSubgraphIndex = 0;

#ifdef TFLMC_XCORE_PROFILE
  printf("\n\nCumulative times for init()...\n");
  for (int i = 0; i < OP_LAST; i++) {
    op_times_summed += op_times[i];
    printf("%-32s %-12d %.2fms\n", op_strs[i], op_times[i], op_times[i]/100000.0);
  }
  printf("\nTotal time for init() - %-10lld %.2fms\n", op_times_summed, op_times_summed/100000.0);
  printf("\n\n\nProfiling prepare()...\n");
  memset(op_times, 0, sizeof(op_times));
  op_times_summed = 0;
#endif

  for(size_t g = 0; g < 1; ++g) {
        currentSubgraphIndex = g;
        for(size_t i = tflNodes_subgraph_index[g]; i < tflNodes_subgraph_index[g+1]; ++i) {
    if (registrations[used_ops[i]].prepare) {

#ifdef TFLMC_XCORE_PROFILE
#if defined(__xcore__) || defined(__riscv_xxcore)
      time_t0 = get_reference_time();
#endif
#endif

      TfLiteStatus status = registrations[used_ops[i]].prepare(&ctx, &tflNodes[i]);

#ifdef TFLMC_XCORE_PROFILE
#if defined(__xcore__) || defined(__riscv_xxcore)
      time_t1 = get_reference_time();
#endif
      op_times[used_ops[i]] += time_t1 - time_t0;
      printf("node %-5d %-32s %-12d\n", i, op_strs[used_ops[i]], time_t1 - time_t0);
#endif

      if (status != kTfLiteOk) {
        return status;
      }
    }
  }
  }
  currentSubgraphIndex = 0;

#ifdef TFLMC_XCORE_PROFILE
  printf("\nCumulative times for prepare()...\n");
  for (int i = 0; i < OP_LAST; i++) {
    op_times_summed += op_times[i];
    printf("%-32s %-12d %.2fms\n", op_strs[i], op_times[i], op_times[i]/100000.0);
  }
  printf("\nTotal time for prepare() - %-10lld %.2fms\n", op_times_summed, op_times_summed/100000.0);
#endif

  return kTfLiteOk;
}

#if defined(__xcore__)
#pragma stackfunction 1000
#endif
TfLiteStatus model_init(void *weights_data_ptr) {
  return model_init_with_paging(weights_data_ptr, nullptr);
}

#if defined(__VX4A__) || defined(__VX4B__)
#define STACKFUNCTION_STATIC(FN, BYTES) \
  asm(".resource_list_empty " # FN ", \"callees\""); \
  asm(".resource_list_empty " # FN ", \"tail_callees\""); \
  asm(".resource_list_empty " # FN ", \"parallel_callees\""); \
  asm(".resource_const " # FN ", \"stack_frame_bytes\", " # BYTES);

STACKFUNCTION_STATIC(_ZN12_GLOBAL__N_117mg_InvokeSubgraphEi, 1000);
STACKFUNCTION_STATIC(_Z10model_initPv, 1024);

#endif

static TfLiteStatus mg_status;
#if defined(__xcore__)
#pragma stackfunction 1000
#endif
__attribute__((fptrgroup("_c_trampoline")))
void model_invoke_subgraph_c_trampoline(){
  mg_status = mg_InvokeSubgraph(0);
}

extern "C" void par_invoke_1(thread_info_t *thread_info, void (*f)());

#if defined(__xcore__)
#pragma stackfunction 1000
#endif
TfLiteStatus model_invoke() {

#ifdef TFLMC_XCORE_PROFILE
  printf("\n\n\nProfiling invoke()...\n");
  memset(op_times, 0, sizeof(op_times));
  memset(op_counts, 0, sizeof(op_counts));
  op_times_summed = 0;
#endif

  par_invoke_1(&xc_config.thread_info, model_invoke_subgraph_c_trampoline);
  if (mg_status != kTfLiteOk) {
    return mg_status;
  }

  #ifdef TFLMC_CONV2D_PROFILE
  struct convopdata{
    const char * name;
    size_t thread_count;
    int evalStartTime;
    int threadsStartTime;
    int threadsDoneTime;
  };
  int conv_times1 = 0, conv_times2 = 0;
  printf("\nConv()...\n");
  for (size_t g = 0; g < 1; ++g) {
    for (size_t i = tflNodes_subgraph_index[g]; i < tflNodes_subgraph_index[g+1]; ++i) {
      if (used_ops[i] == OP_XC_conv2d_v2) {
        auto *op_data = reinterpret_cast<convopdata *>(tflNodes[i].user_data);
        conv_times1 += op_data->threadsStartTime - op_data->evalStartTime;
        conv_times2 += op_data->threadsDoneTime - op_data->threadsStartTime;
        printf("node %-5d %-25s %-25s %-6d %-6d %-12d\n", i, op_strs[used_ops[i]], op_data->name, op_data->thread_count, op_data->threadsStartTime - op_data->evalStartTime, op_data->threadsDoneTime - op_data->threadsStartTime);
      }
    }
  }
  printf("Summed - %-10d %-10d\n", conv_times1, conv_times2);
#endif
    
#ifdef TFLMC_XCORE_PROFILE
  printf("\nCumulative times for invoke()...\n");
  for (int i = 0; i < OP_LAST; i++) {
    op_times_summed += op_times[i];
    printf("%-5d %-32s %-12d %.2fms\n", op_counts[i], op_strs[i], op_times[i], op_times[i]/100000.0);
  }
  printf("\nTotal time for invoke() - %-10lld %.2fms\n", op_times_summed, op_times_summed/100000.0);
#endif

  return kTfLiteOk;
}

TfLiteStatus model_reset() {
  // Reset variable tensors
  for (int i = 0; i < 0; i++) {
    memset(tflTensors[varTensors_index[i]].data.data, tflTensors[varTensors_index[i]].params.zero_point, TensorBytes(&tflTensors[varTensors_index[i]]));
  }
  return kTfLiteOk;
}

#if defined(__xcore__) && defined(USB_TILE)
#include "ioserver.h"
#include <xcore/hwtimer.h>
extern "C" {
extern int read_sswitch_reg(unsigned tile, unsigned reg, unsigned *data);
extern int write_sswitch_reg(unsigned tile, unsigned reg, unsigned data);
}

#pragma stackfunction 1000
void model_ioserver(chanend_t c) {
    unsigned tensor_num = 0;
    extern unsigned tile[];
    while(1) {
        int cmd = ioserver_command_receive(c, &tensor_num);
        switch(cmd) {
        case IOSERVER_TENSOR_RECV_INPUT: {
            ioserver_tensor_recv_input(
                c, (unsigned int *) model_input(tensor_num)->data.u32,
                (model_input_size(tensor_num) + 3) / sizeof(int));
            break;
        }
        case IOSERVER_TENSOR_SEND_OUTPUT: {
            ioserver_tensor_send_output(
                c, (unsigned int*) model_output(tensor_num)->data.u32, 
                (model_output_size(tensor_num) + 3) / sizeof(int));
            break;
        }
        case IOSERVER_INVOKE: {
            model_invoke();
            ioserver_command_acknowledge(c, IOSERVER_ACK);
            break;
        }
        case IOSERVER_RESET: {
            model_reset();
            ioserver_command_acknowledge(c, IOSERVER_ACK);
            break;
        }
        case IOSERVER_EXIT: {
          ioserver_command_acknowledge(c, IOSERVER_ACK);
          unsigned pll_ctrl;
          hwtimer_t timer = hwtimer_alloc();
          hwtimer_delay(timer, 100000);
          hwtimer_free(timer);
          read_sswitch_reg(USB_TILE, XS1_SSWITCH_PLL_CTL_NUM, &pll_ctrl);
          write_sswitch_reg(USB_TILE, XS1_SSWITCH_PLL_CTL_NUM, pll_ctrl);
          return;
        }
        default: {
            ioserver_command_acknowledge(c, IOSERVER_NACK);
            break;
        }
        }
    }
}
#else 

void model_ioserver(void *io_channel) {}

#endif // __xcore__

