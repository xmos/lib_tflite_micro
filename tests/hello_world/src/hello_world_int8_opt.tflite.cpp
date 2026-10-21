// This file is generated. Do not edit.
// Generated on: 21.10.2026 16:31:20
// Compiler version: 1.4.3.dev50
// Args: hello_world_int8.tflite -o hello_world_int8_opt.tflite 

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
extern TFLMRegistration *Register_XC_no_op(void);
extern TFLMRegistration *Register_XC_pad_1_to_4(void);
extern TFLMRegistration *Register_XC_conv2d_v2(void);
extern TFLMRegistration *Register_XC_slice(void);
} // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite_micro



constexpr int kTensorArenaSize = 392;
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
  OP_XC_no_op, OP_XC_pad_1_to_4, OP_XC_conv2d_v2, OP_XC_slice,  OP_LAST
};

#if defined(TFLMC_XCORE_PROFILE) || defined(TFLMC_PRINT_TENSORS) || defined(TFLMC_PRINT_INPUT_TENSORS) || defined(TFLMC_CONV2D_PROFILE)
const char *op_strs[] = {
"OP_XC_no_op", "OP_XC_pad_1_to_4", "OP_XC_conv2d_v2", "OP_XC_slice", };

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
const TfArray<2, int> tensor_dimension0 = { 2, { 1,1 } };
const TfArray<1, float> quant0_scale = { 1, { 0.024480115622282028, } };
const TfArray<1, int> quant0_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant0 = { (TfLiteFloatArray*)&quant0_scale, (TfLiteIntArray*)&quant0_zero, 0 };
const ALIGN(8) int16_t tensor_data1[32] = { 
    31123, 0, 31123, 31123, 31123, 31123, 31123, 31123, 31123, 31123, 
    31123, 31123, 0, 31123, 31123, 31123, -17479, -16385, -14620, -12991, 
    -5026, -3829, -8414, -4429, -5379, -19546, 936, -18646, -16385, -14732, 
    -1600, -12784, 
};
const TfArray<1, int> tensor_dimension1 = { 1, { 32 } };
const ALIGN(8) int8_t tensor_data2[512] = { 
    86, 0, 0, 0, 95, 0, 0, 0, 68, 0, 
    0, 0, -82, 0, 0, 0, 25, 0, 0, 0, 
    127, 0, 0, 0, -26, 0, 0, 0, 64, 0, 
    0, 0, 99, 0, 0, 0, 98, 0, 0, 0, 
    115, 0, 0, 0, 104, 0, 0, 0, 71, 0, 
    0, 0, 57, 0, 0, 0, -54, 0, 0, 0, 
    -9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 
};
const TfArray<1, int> tensor_dimension2 = { 1, { 512 } };
const ALIGN(8) int16_t tensor_data3[32] = { 
    11916, 11916, 11916, 11916, 11916, 11916, 11916, 11916, 11916, 11916, 
    11916, 11916, 11916, 11916, 11916, 11916, -4316, -11811, -2805, -11684, 
    -3404, 1550, -8578, -1713, 3651, -1046, -11578, -12717, -4999, -8286, 
    -3024, 8942, 
};
const ALIGN(8) int8_t tensor_data4[512] = { 
    8, 34, 38, 33, 9, 8, 15, 11, -32, 18, 
    -12, 127, -36, 88, -27, 38, -23, 36, -26, 41, 
    0, 7, 22, -30, 30, 13, 35, -45, -35, -9, 
    20, -6, 25, 30, -30, 9, -3, -13, 20, -35, 
    -38, 32, -39, 15, -29, -7, -9, -18, -37, -25, 
    21, 21, -11, 6, -42, 26, -36, 9, 34, -2, 
    8, 2, 19, -17, -30, 31, -23, -41, -54, -30, 
    -35, -58, 19, -25, 4, 62, 0, 1, 20, -57, 
    -22, 26, -4, -40, 19, 0, -60, -40, -20, -39, 
    -2, 13, 25, 32, -40, -42, -17, 28, -25, 3, 
    -32, 22, 2, 3, 33, 24, 9, 46, -39, -27, 
    20, 11, 37, 33, 29, 10, -37, 29, -36, 32, 
    1, -6, -29, 55, 11, -15, 26, 22, -35, -5, 
    -3, 0, 20, 38, 17, 23, -25, -15, 17, -22, 
    2, 38, 4, 4, 17, 12, -11, -32, -13, 7, 
    23, -27, -24, -19, -6, -36, -24, 35, -5, 7, 
    35, 10, 34, 36, 30, 39, 3, -26, 3, 36, 
    -1, -64, 17, -8, -4, -15, -28, 21, 7, 11, 
    4, 27, 26, -2, -21, 1, -34, 33, -26, 11, 
    -20, 3, 16, -2, 20, -38, 28, -8, -13, -15, 
    -17, -30, -13, 9, -29, -23, -19, -29, -5, 39, 
    -35, -21, -37, -28, 5, 26, 23, -4, 36, 18, 
    21, -17, 30, -28, 3, -36, -46, 2, 6, -7, 
    -12, 2, -1, -6, -17, -15, -17, -45, 39, -31, 
    -12, 26, -19, 9, 25, 33, -12, 36, -32, 33, 
    -17, -68, -9, -11, -6, 25, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 
};
const ALIGN(8) int16_t tensor_data5[20] = { 
    24877, 0, 0, 0, 11077, 319, 319, 319, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
};
const TfArray<1, int> tensor_dimension5 = { 1, { 20 } };
const ALIGN(8) int8_t tensor_data6[512] = { 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, -39, 59, 
    39, 21, 28, -32, -34, -35, 15, 27, -59, -41, 
    18, -35, -7, 127, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 
};
const TfArray<4, int> tensor_dimension7 = { 4, { 1,1,1,1 } };
const TfArray<4, int> tensor_dimension8 = { 4, { 1,1,1,4 } };
const TfArray<1, int> tensor_dimension9 = { 1, { 64 } };
const TfArray<4, int> tensor_dimension10 = { 4, { 1,1,1,16 } };
const TfArray<1, float> quant10_scale = { 1, { 0.01332512404769659, } };
const TfArray<1, int> quant10_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant10 = { (TfLiteFloatArray*)&quant10_scale, (TfLiteIntArray*)&quant10_zero, 0 };
const TfArray<1, float> quant12_scale = { 1, { 0.012775269336998463, } };
const TfArray<1, int> quant12_zero = { 1, { -128 } };
const TfLiteAffineQuantization quant12 = { (TfLiteFloatArray*)&quant12_scale, (TfLiteIntArray*)&quant12_zero, 0 };
const TfArray<1, float> quant14_scale = { 1, { 0.0082909567281603813, } };
const TfArray<1, int> quant14_zero = { 1, { 5 } };
const TfLiteAffineQuantization quant14 = { (TfLiteFloatArray*)&quant14_scale, (TfLiteIntArray*)&quant14_zero, 0 };
uint8_t ALIGN(4) opdata0[0] = {  }; /* custom_initial_data */
const TfArray<1, int> inputs0 = { 1, { 0 } };
const TfArray<1, int> outputs0 = { 1, { 7 } };
uint8_t ALIGN(4) opdata1[28] = { 112, 118, 0, 1, 4, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 128, 128, 128, 128, 6, 5, 38, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs1 = { 1, { 7 } };
const TfArray<1, int> outputs1 = { 1, { 8 } };
uint8_t ALIGN(4) opdata2[152] = { 109, 112, 0, 40, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 252, 255, 255, 255, 228, 255, 255, 255, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 97, 0, 8, 16, 0, 0, 0, 4, 0, 0, 0, 0, 111, 0, 8, 16, 0, 0, 0, 1, 0, 255, 255, 0, 112, 0, 42, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 80, 6, 127, 71, 60, 12, 9, 7, 1, 7, 87, 1, 133, 78, 23, 64, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs2 = { 5, { 8,2,1,-1,9 } };
const TfArray<1, int> outputs2 = { 1, { 10 } };
uint8_t ALIGN(4) opdata3[152] = { 109, 112, 0, 40, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 255, 255, 255, 240, 255, 255, 255, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 97, 0, 8, 16, 0, 0, 0, 16, 0, 0, 0, 0, 111, 0, 8, 16, 0, 0, 0, 1, 0, 253, 255, 0, 112, 0, 42, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 80, 6, 127, 71, 60, 12, 9, 7, 1, 7, 87, 1, 133, 78, 23, 64, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs3 = { 5, { 10,4,3,-1,11 } };
const TfArray<1, int> outputs3 = { 1, { 12 } };
uint8_t ALIGN(4) opdata4[152] = { 109, 112, 0, 40, 16, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 240, 255, 255, 255, 240, 255, 255, 255, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 97, 0, 8, 4, 0, 0, 0, 16, 0, 0, 0, 0, 111, 0, 8, 4, 0, 0, 0, 0, 0, 254, 255, 0, 112, 0, 42, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 48, 48, 0, 1, 44, 20, 115, 0, 107, 0, 116, 0, 7, 80, 6, 127, 71, 60, 12, 9, 7, 1, 7, 87, 1, 133, 78, 23, 64, 0, 20, 4, 20, 20, 40, 4, 4, 14, 36, 1,  }; /* custom_initial_data */
const TfArray<5, int> inputs4 = { 5, { 12,6,5,-1,13 } };
const TfArray<1, int> outputs4 = { 1, { 14 } };
uint8_t ALIGN(4) opdata5[32] = { 115, 0, 111, 0, 108, 0, 110, 0, 118, 0, 5, 7, 6, 11, 14, 7, 5, 1, 5, 1, 1, 4, 0, 0, 4, 4, 4, 4, 104, 10, 36, 1,  }; /* custom_initial_data */
const TfArray<1, int> inputs5 = { 1, { 14 } };
const TfArray<1, int> outputs5 = { 1, { 15 } };
const TfArray<1, int> inputs6 = { 1, { 15 } };
const TfArray<1, int> outputs6 = { 1, { 16 } };
} g0;

TfLiteTensor tflTensors[] = 
{{ {(int32_t*)(tensor_arena + 72)},(TfLiteIntArray*)&g0.tensor_dimension0, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant0)) }, {g0.quant0.scale->data[0], g0.quant0.zero_point->data[0] },},
{ {(int32_t*)g0.tensor_data1},(TfLiteIntArray*)&g0.tensor_dimension1, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)g0.tensor_data2},(TfLiteIntArray*)&g0.tensor_dimension2, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)g0.tensor_data3},(TfLiteIntArray*)&g0.tensor_dimension1, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)g0.tensor_data4},(TfLiteIntArray*)&g0.tensor_dimension2, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)g0.tensor_data5},(TfLiteIntArray*)&g0.tensor_dimension5, kTfLiteInt16, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)g0.tensor_data6},(TfLiteIntArray*)&g0.tensor_dimension2, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 72)},(TfLiteIntArray*)&g0.tensor_dimension7, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant0)) }, {g0.quant0.scale->data[0], g0.quant0.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 64)},(TfLiteIntArray*)&g0.tensor_dimension8, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant0)) }, {g0.quant0.scale->data[0], g0.quant0.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension9, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 80)},(TfLiteIntArray*)&g0.tensor_dimension10, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant10)) }, {g0.quant10.scale->data[0], g0.quant10.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension9, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 64)},(TfLiteIntArray*)&g0.tensor_dimension10, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant12)) }, {g0.quant12.scale->data[0], g0.quant12.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension9, kTfLiteInt8, {kTfLiteNoQuantization, nullptr }, {0,0},},
{ {(int32_t*)(tensor_arena + 80)},(TfLiteIntArray*)&g0.tensor_dimension8, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant14)) }, {g0.quant14.scale->data[0], g0.quant14.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension7, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant14)) }, {g0.quant14.scale->data[0], g0.quant14.zero_point->data[0] },},
{ {(int32_t*)(tensor_arena + 0)},(TfLiteIntArray*)&g0.tensor_dimension0, kTfLiteInt8, {kTfLiteAffineQuantization, const_cast<void*>(static_cast<const void*>(&g0.quant14)) }, {g0.quant14.scale->data[0], g0.quant14.zero_point->data[0] },},
};

TfLiteNode tflNodes[] = 
{{ (TfLiteIntArray*)&g0.inputs0, (TfLiteIntArray*)&g0.outputs0, (TfLiteIntArray*)&g0.inputs0, const_cast<void*>(static_cast<const void*>(&g0.opdata0)), 0, },
{ (TfLiteIntArray*)&g0.inputs1, (TfLiteIntArray*)&g0.outputs1, (TfLiteIntArray*)&g0.inputs1, const_cast<void*>(static_cast<const void*>(&g0.opdata1)), 28, },
{ (TfLiteIntArray*)&g0.inputs2, (TfLiteIntArray*)&g0.outputs2, (TfLiteIntArray*)&g0.inputs2, const_cast<void*>(static_cast<const void*>(&g0.opdata2)), 152, },
{ (TfLiteIntArray*)&g0.inputs3, (TfLiteIntArray*)&g0.outputs3, (TfLiteIntArray*)&g0.inputs3, const_cast<void*>(static_cast<const void*>(&g0.opdata3)), 152, },
{ (TfLiteIntArray*)&g0.inputs4, (TfLiteIntArray*)&g0.outputs4, (TfLiteIntArray*)&g0.inputs4, const_cast<void*>(static_cast<const void*>(&g0.opdata4)), 152, },
{ (TfLiteIntArray*)&g0.inputs5, (TfLiteIntArray*)&g0.outputs5, (TfLiteIntArray*)&g0.inputs5, const_cast<void*>(static_cast<const void*>(&g0.opdata5)), 32, },
{ (TfLiteIntArray*)&g0.inputs6, (TfLiteIntArray*)&g0.outputs6, (TfLiteIntArray*)&g0.inputs6, const_cast<void*>(static_cast<const void*>(&g0.opdata0)), 0, },
};

used_operators_e used_ops[] =
{OP_XC_no_op, OP_XC_pad_1_to_4, OP_XC_conv2d_v2, OP_XC_conv2d_v2, OP_XC_conv2d_v2, OP_XC_slice, OP_XC_no_op, };


// Indices into tflTensors and tflNodes for subgraphs
size_t tflTensors_subgraph_index[] = {0, 17, };
size_t tflNodes_subgraph_index[] = {0, 7, };

// Variable tensors
size_t varTensors_index[] = {};

// Input/output tensors
static const int inTensorIndices[] = {
  0, 
};

static const int outTensorIndices[] = {
  16, 
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
  ctx.tensors_size = 17;
  registrations[OP_XC_no_op] = *(tflite_micro::ops::micro::xcore::Register_XC_no_op());
  registrations[OP_XC_pad_1_to_4] = *(tflite_micro::ops::micro::xcore::Register_XC_pad_1_to_4());
  registrations[OP_XC_conv2d_v2] = *(tflite_micro::ops::micro::xcore::Register_XC_conv2d_v2());
  registrations[OP_XC_slice] = *(tflite_micro::ops::micro::xcore::Register_XC_slice());


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

