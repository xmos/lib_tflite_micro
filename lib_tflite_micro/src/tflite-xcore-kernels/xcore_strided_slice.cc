// Copyright (c) 2022, XMOS Ltd, All rights reserved


#include "MemCpyFn.hpp"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "xcore_custom_options.h"
#include "xcore_interpreter.h"
#include "xcore_dispatcher.h"
#include "xcore_utils.h"
extern "C" {
#include "nn_operator.h"
}
#include <cstdio>
#include <iostream>


namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace strided_slice {

// This is the struct that contains the data required by the operator
struct StridedSliceOpData : XCoreOpData {   // Inherits the operator name field from XCoreOpData
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t begin_x;
    uint32_t begin_y;
    uint32_t end_x;
    uint32_t end_y;
    uint32_t stride_x;
    uint32_t stride_y;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto op_data = construct_persistent_object<StridedSliceOpData>(context);
  op_data->name = "XC_Strided_Slice";

  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = static_cast<StridedSliceOpData*>(node->user_data);

  //Get Inputs and set op data
  const TfLiteTensor* input_ten = GetInput(context, node, 0);
  const TfLiteTensor* begin_ten = GetInput(context, node, 1);
  const TfLiteTensor* end_ten = GetInput(context, node, 2);
  const TfLiteTensor* strides_ten = GetInput(context, node, 3);
  
  op_data->width = SizeOfDimension(input_ten, 2);
  
  op_data->height = SizeOfDimension(input_ten, 1);
  op_data->channels = SizeOfDimension(input_ten, 3);  

  const uint32_t *begins = GetTensorData<uint32_t>(begin_ten);
  op_data->begin_x = begins[2];
  op_data->begin_y = begins[1];
  if(!(op_data->begin_x < op_data->width)){
    op_data->begin_x = op_data->width;
  }
  if(!(op_data->begin_y < op_data->height)){
    op_data->begin_y = op_data->height;
  }

  const uint32_t *ends = GetTensorData<uint32_t>(end_ten);
  op_data->end_x = ends[2];
  op_data->end_y = ends[1];
  if(!(op_data->end_x < op_data->width)){
    op_data->end_x = op_data->width;
  }
  if(!(op_data->end_y < op_data->height)){
    op_data->end_y = op_data->height;
  }

  const uint32_t *strides = GetTensorData<uint32_t>(strides_ten);
  op_data->stride_x = strides[2];
  op_data->stride_y = strides[1];

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = static_cast<StridedSliceOpData*>(node->user_data);

  //Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  //Pointers to data in In/Out Tensors
  void* in_data = const_cast<void *>(tflite::micro::GetTensorData<void>(input));
  void* out_data = tflite::micro::GetTensorData<void>(output);

  uint8_t* output_iter = (uint8_t*)out_data;

  for(uint32_t row_iter{0}; row_iter < (op_data->end_y - op_data->begin_y); row_iter += op_data->stride_y)
  {
    uint8_t* input_iter = ((uint8_t*)in_data) + (op_data->begin_x + (op_data->begin_y + row_iter)*op_data->width)*op_data->channels;
    for(uint32_t col_iter{0}; col_iter < (op_data->end_x - op_data->begin_x); col_iter += op_data->stride_x)
    {
      memcpy(output_iter, input_iter, op_data->channels);
      output_iter += op_data->channels;
      input_iter += op_data->stride_x*op_data->channels;
    }
  }
  return kTfLiteOk;
}

}  // namespace strided_slice


TfLiteRegistration *Register_Strided_Slice () {
  static TfLiteRegistration r = {strided_slice::Init, nullptr, strided_slice::Prepare,
                                 strided_slice::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite