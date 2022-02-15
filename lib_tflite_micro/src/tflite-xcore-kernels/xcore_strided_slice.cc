// Copyright (c) 2022, XMOS Ltd, All rights reserved

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

#define TEST

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
  TFLITE_DCHECK(buffer != nullptr);
  
  auto op_data = construct_persistent_object<StridedSliceOpData>(context);
  auto parser = CustomOptionParser(buffer, length);
  op_data->width = parser.parseNamedCustomOption("width").AsInt32();
  op_data->height = parser.parseNamedCustomOption("height").AsInt32();
  op_data->channels = parser.parseNamedCustomOption("channels").AsInt32();
  op_data->begin_x = parser.parseNamedCustomOption("begin_x").AsInt32();
  op_data->begin_x = parser.parseNamedCustomOption("begin_y").AsInt32();
  op_data->end_x = parser.parseNamedCustomOption("end_x").AsInt32();
  op_data->end_x = parser.parseNamedCustomOption("end_y").AsInt32();
  op_data->stride_x = parser.parseNamedCustomOption("stride_x").AsInt32();
  op_data->stride_y = parser.parseNamedCustomOption("stride_y").AsInt32();
  op_data->name = "XC_Strided_Slice";

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


  #ifdef TEST
    //Force Input Tensor
    int32_t set_channel_count{0};
    int8_t setVal{0};
    int8_t* setPtr = ((int8_t*)in_data);
    for(int i{0}; i<(op_data->width*op_data->height*op_data->channels);i++){
      if(set_channel_count%op_data->channels == 0) setVal++;
      set_channel_count++;
      setPtr[i] = setVal;
    }
  #endif

  uint8_t* output_iter = (uint8_t*)out_data;

  for(uint32_t row_iter{0}; row_iter <= (op_data->end_y - op_data->begin_y); row_iter += op_data->stride_y)
  {
    uint8_t* input_iter = ((uint8_t*)in_data) + (op_data->begin_x + (op_data->begin_y + row_iter)*op_data->width)*op_data->channels;
    for(uint32_t col_iter{0}; col_iter <= (op_data->end_x - op_data->begin_x); col_iter += op_data->stride_x)
    {
      memcpy(output_iter, input_iter, op_data->channels);
      output_iter += op_data->channels;
      input_iter += op_data->stride_x*op_data->channels;
    }
  }
  
  #ifdef TEST
    printf("\n\nDimensions: %d %d %d \n", op_data->width, op_data->height, op_data->channels);
    printf("Begin coords\n");
    printf("%d %d\n", op_data->begin_x, op_data->begin_y);
    printf("End coords\n");
    printf("%d %d\n", op_data->end_x, op_data->end_y);
    printf("Stride\n");
    printf("%d %d\n", op_data->stride_x, op_data->stride_y);

    printf("\n\nInput Data\n");
    int32_t channel_count{0};
    int8_t* intPtr = ((int8_t*)in_data);
    for(int i{0}; i<(op_data->width*op_data->height*op_data->channels);i++){
      if(channel_count%op_data->channels == 0) printf("\n");
      channel_count++;
      printf("%d, ", intPtr[i]);
    }
    
    channel_count = 0;
    printf("\n\nOutput Data\n");
    int8_t* outPtr = ((int8_t*)out_data);
    for(int i{0}; i< ((op_data->end_x-op_data->begin_x+2-op_data->stride_x)*(op_data->end_y-op_data->begin_y+2-op_data->stride_y)*op_data->channels);i++){
      if(channel_count%op_data->channels == 0) printf("\n");
      channel_count++;
      printf("%d, ", outPtr[i]);
    }
  #endif

  return kTfLiteOk;
}

}  // namespace strided_slice

TfLiteRegistration *Register_Strided_Slice() {
  static TfLiteRegistration r = {strided_slice::Init, nullptr, strided_slice::Prepare,
                                 strided_slice::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite