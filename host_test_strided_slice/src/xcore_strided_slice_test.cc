/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdint>
#include <iostream>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

#include "xcore_ops.h"

#define MEM_OVERREAD 32

namespace tflite {
namespace testing {
namespace {

template <typename T>
void ValidateStridedSliceGoldens(TfLiteTensor* tensors, int tensors_size,
                                 const T* golden, T* output, int output_len,
                                 TfLiteStridedSliceParams* params,
                                 const bool expect_prepare_err, int num_invoke,
                                 float tolerance = 1e-5) {
  int inputs_array_data[] = {4, 0, 1, 2, 3};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 4};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  TfLiteRegistration* registration =
      tflite::ops::micro::xcore::Register_Strided_Slice_V3();
  micro::KernelRunner runner(*registration, tensors, tensors_size, inputs_array,
                             outputs_array, reinterpret_cast<void*>(params));
  if (expect_prepare_err) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteError, runner.InitAndPrepare());
    return;
  } else {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  }

  for (int i = 0; i < num_invoke; i++) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
  }

  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output[i], 1e-5f);
  }
}


template <typename T>
void TestStridedSliceQuantized(int* input_shape, int* begin_shape,
                               int* end_shape, int* strides_shape,
                               TfLiteStridedSliceParams* builtin_data,
                               const T* input_data, const int32_t* begin_data,
                               const int32_t* end_data,
                               const int32_t* strides_data, int* output_shape,
                               T* output_data, const T* expected_output,
                               bool expect_prepare_err, int num_invoke = 1) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_shape);
  TfLiteIntArray* begin_dims = IntArrayFromInts(begin_shape);
  TfLiteIntArray* end_dims = IntArrayFromInts(end_shape);
  TfLiteIntArray* strides_dims = IntArrayFromInts(strides_shape);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_shape);
  constexpr int inputs_size = 4;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  int zero_point =
      std::numeric_limits<T>::max() + std::numeric_limits<T>::min() / 2;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, 1.0, zero_point),
      CreateTensor(begin_data, begin_dims),
      CreateTensor(end_data, end_dims),
      CreateTensor(strides_data, strides_dims),
      CreateQuantizedTensor(output_data, output_dims, 1.0, zero_point),
  };
  ValidateStridedSliceGoldens(tensors, tensors_size, expected_output,
                              output_data, ElementCount(*output_dims),
                              builtin_data, expect_prepare_err, num_invoke,
                              1.0);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(ShrinkXY) {
  int input_shape[] = {4, 1, 3, 3, 4};
  int begin_shape[] = {2, 1, 3};
  int end_shape[] = {2, 1, 3};
  int strides_shape[] = {2, 1, 3};
  int output_shape[] = {3, 2, 2, 4};
  int8_t input_data[64];
  for(int i{0}; i < 36; i++)
  {
    input_data[i] = i+1;
  }
  
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {0, 2, 2};
  int32_t strides_data[] = {0, 1, 1};
  int8_t golden[] = {1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 19, 20};
  int8_t output_data[MEM_OVERREAD*2];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(ShrinkX) {
  int input_shape[] = {4, 1, 3, 3, 4};
  int begin_shape[] = {2, 1, 3};
  int end_shape[] = {2, 1, 3};
  int strides_shape[] = {2, 1, 3};
  int output_shape[] = {3, 1, 3, 4};
  int8_t input_data[64];
  for(int i{0}; i < 36; i++)
  {
    input_data[i] = i+1;
  }
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {0, 1, 3};
  int32_t strides_data[] = {0, 1, 1};
  int8_t golden[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int8_t output_data[2*MEM_OVERREAD];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(ShrinkY) {
  int input_shape[] = {4, 1, 3, 3, 4};
  int begin_shape[] = {2, 1, 3};
  int end_shape[] = {2, 1, 3};
  int strides_shape[] = {2, 1, 3};
  int output_shape[] = {3, 3, 1, 4};
  int8_t input_data[64];
  for(int i{0}; i < 36; i++)
  {
    input_data[i] = i+1;
  }
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {0, 3, 1};
  int32_t strides_data[] = {0, 1, 1};
  int8_t golden[] = {1, 2, 3, 4, 13, 14, 15, 16, 25, 26, 27, 28};
  int8_t output_data[2*MEM_OVERREAD];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(CenterXY) {
  int input_shape[] = {4, 1, 3, 3, 4};
  int begin_shape[] = {2, 1, 3};
  int end_shape[] = {2, 1, 3};
  int strides_shape[] = {2, 1, 3};
  int output_shape[] = {3, 1, 1, 4};
  int8_t input_data[64];
  for(int i{0}; i < 36; i++)
  {
    input_data[i] = i+1;
  }
  int32_t begin_data[] = {0, 1, 2};
  int32_t end_data[] = {0, 2, 3};
  int32_t strides_data[] = {0, 1, 1};
  int8_t golden[] = {21, 22, 23, 24};
  int8_t output_data[2*MEM_OVERREAD];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(CenterX) {
  int input_shape[] = {4, 1, 3, 3, 4};
  int begin_shape[] = {2, 1, 3};
  int end_shape[] = {2, 1, 3};
  int strides_shape[] = {2, 1, 3};
  int output_shape[] = {3, 3, 1, 4};
  int8_t input_data[64];
  for(int i{0}; i < 36; i++)
  {
    input_data[i] = i+1;
  }
  int32_t begin_data[] = {0, 0, 1};
  int32_t end_data[] = {0, 3, 2};
  int32_t strides_data[] = {0, 1, 1};
  int8_t golden[] = {5, 6, 7, 8, 17, 18, 19, 20, 29, 30, 31, 32};
  int8_t output_data[2*MEM_OVERREAD];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(CenterY) {
  int input_shape[] = {4, 1, 3, 3, 4};
  int begin_shape[] = {2, 1, 3};
  int end_shape[] = {2, 1, 3};
  int strides_shape[] = {2, 1, 3};
  int output_shape[] = {3, 1, 3, 4};
  int8_t input_data[64];
  for(int i{0}; i < 36; i++)
  {
    input_data[i] = i+1;
  }
  int32_t begin_data[] = {0, 1, 0};
  int32_t end_data[] = {0, 2, 3};
  int32_t strides_data[] = {0, 1, 1};
  int8_t golden[] = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  int8_t output_data[2*MEM_OVERREAD];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(StridesXY) {
  int input_shape[] = {4, 1, 3, 3, 4};
  int begin_shape[] = {2, 1, 3};
  int end_shape[] = {2, 1, 3};
  int strides_shape[] = {2, 1, 3};
  int output_shape[] = {3, 2, 2, 4};
  int8_t input_data[64];
  for(int i{0}; i < 36; i++)
  {
    input_data[i] = i+1;
  }
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {0, 3, 3};
  int32_t strides_data[] = {0, 2, 2};
  int8_t golden[] = {1, 2, 3, 4, 9, 10, 11, 12, 25, 26, 27, 28, 33, 34, 35, 36};
  int8_t output_data[2*MEM_OVERREAD];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(StridesX) {
  int input_shape[] = {4, 1, 3, 3, 4};
  int begin_shape[] = {2 ,1, 3};
  int end_shape[] = {2, 1, 3};
  int strides_shape[] = {2, 1, 3};
  int output_shape[] = {3, 2, 3, 4};
  int8_t input_data[64];
  for(int i{0}; i < 36; i++)
  {
    input_data[i] = i+1;
  }
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {0, 3, 3};
  int32_t strides_data[] = {0, 2, 1};
  int8_t golden[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34 ,35, 36};
  int8_t output_data[2*MEM_OVERREAD];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(StridesY) {
  int input_shape[] = {4, 1, 3, 3, 4};
  int begin_shape[] = {2, 1, 3};
  int end_shape[] = {2, 1, 3};
  int strides_shape[] = {2, 1, 3};
  int output_shape[] = {3, 3, 2, 4};
  int8_t input_data[64];
  for(int i{0}; i < 36; i++)
  {
    input_data[i] = i+1;
  }
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {0, 3, 3};
  int32_t strides_data[] = {0, 1, 2};
  int8_t golden[] = {1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 27, 28, 33, 34, 35, 36};
  int8_t output_data[2*MEM_OVERREAD];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(BigShrinkXY_StrideXY) {
  int input_shape[] = {4, 1, 5, 5, 4};
  int begin_shape[] = {2, 1, 3};
  int end_shape[] = {2, 1, 3};
  int strides_shape[] = {2, 1, 3};
  int output_shape[] = {3, 2, 2, 4};
  int8_t input_data[128];
  for(int i{0}; i < 100; i++)
  {
    input_data[i] = i+1;
  }
  int32_t begin_data[] = {0, 1, 2};
  int32_t end_data[] = {0, 4, 5};
  int32_t strides_data[] = {0, 2, 2};
  int8_t golden[] = {29, 30, 31, 32, 37, 38, 39, 40, 69, 70, 71, 72, 77, 78, 79, 80};
  int8_t output_data[4*MEM_OVERREAD];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(BigUnevenShrinkXY) {
  int input_shape[] = {4, 1, 5, 5, 4};
  int begin_shape[] = {2, 1, 3};
  int end_shape[] = {2, 1, 3};
  int strides_shape[] = {2, 1, 3};
  int output_shape[] = {3, 3, 2, 4};
  int8_t input_data[128];
  for(int i{0}; i < 100; i++)
  {
    input_data[i] = i+1;
  }
  int32_t begin_data[] = {0, 2, 1};
  int32_t end_data[] = {0, 5, 3};
  int32_t strides_data[] = {0, 1, 1};
  int8_t golden[] = {45, 46, 47, 48, 49, 50, 51, 52, 65, 66, 67, 68, 69, 70, 71, 72, 85, 86, 87, 88, 89, 90, 91, 92};
  int8_t output_data[4*MEM_OVERREAD];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(BigUnevenStrideXY) {
  int input_shape[] = {4, 1, 5, 5, 4};
  int begin_shape[] = {2, 1, 3};
  int end_shape[] = {2, 1, 3};
  int strides_shape[] = {2, 1, 3};
  int output_shape[] = {3, 2, 3, 4};
  int8_t input_data[200];
  for(int i{0}; i < 100; i++)
  {
    input_data[i] = i+1;
  }
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {0, 5, 5};
  int32_t strides_data[] = {0, 3, 2};
  int8_t golden[] = {1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19, 20, 61, 62, 63, 64, 69, 70, 71, 72, 77, 78, 79, 80};
  int8_t output_data[6*MEM_OVERREAD];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(FAILCASE) {
  int input_shape[] = {4, 1, 3, 3, 4};
  int begin_shape[] = {2, 1, 3};
  int end_shape[] = {2, 1, 3};
  int strides_shape[] = {2, 1, 3};
  int output_shape[] = {3, 2, 2, 4};
  int8_t input_data[64];
  for(int i{0}; i < 36; i++)
  {
    input_data[i] = i+1;
  }
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {0, 2, 2};
  int32_t strides_data[] = {0, 1, 1};
  int8_t golden[] = {1, 2, 3, 6, 7, 8, 9, 11, 1, 2, 3, 6, 7, 8, 9, 11};
  int8_t output_data[2*MEM_OVERREAD];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, true);
}

TF_LITE_MICRO_TESTS_END
