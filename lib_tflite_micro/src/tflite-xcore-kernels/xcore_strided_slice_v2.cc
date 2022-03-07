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

//#define TEST
//#define DEBUG

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace strided_slice_v2 {

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

  	TFLITE_DCHECK(buffer != nullptr);
	auto parser = CustomOptionParser(buffer, length);
	op_data->width = parser.parseNamedCustomOption("width").AsInt32();
	op_data->height = parser.parseNamedCustomOption("height").AsInt32();
	op_data->channels = parser.parseNamedCustomOption("channels").AsInt32();
	op_data->begin_x = parser.parseNamedCustomOption("begin_x").AsInt32();
	op_data->begin_y = parser.parseNamedCustomOption("begin_y").AsInt32();
	op_data->end_x = parser.parseNamedCustomOption("end_x").AsInt32();
	op_data->end_y = parser.parseNamedCustomOption("end_y").AsInt32();
	op_data->stride_x = parser.parseNamedCustomOption("stride_x").AsInt32();
	op_data->stride_y = parser.parseNamedCustomOption("stride_y").AsInt32();

  	op_data->name = "XC_Strided_Slice_V2";

  	return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
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

}  // namespace strided_slice_v2


TfLiteRegistration *Register_Strided_Slice_V2 () {
  static TfLiteRegistration r = {strided_slice_v2::Init, nullptr, strided_slice_v2::Prepare,
                                 strided_slice_v2::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
