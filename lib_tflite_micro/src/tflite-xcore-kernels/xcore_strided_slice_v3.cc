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
namespace strided_slice_v3 {

// This is the struct that contains the data required by the operator
struct StridedSliceOpData : XCoreOpData {   // Inherits the operator name field from XCoreOpData
    nn::ImToColValid::Params *mf_params;
};

template <typename T>
T* getDeserializedParams(TfLiteContext* context, const uint8_t* data) {
  char* allocated_memory;
  int allocationByteCount = sizeof(T);
  allocated_memory =
      (char*)context->AllocatePersistentBuffer(context, allocationByteCount);
  T* param = T::template deserialise<T>(allocated_memory, (const char*)data);
  return param;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto op_data = construct_persistent_object<StridedSliceOpData>(context);
  op_data->name = "XC_Strided_Slice";
  auto parser = CustomOptionParser(buffer, length);
  const uint8_t *memcpy_fn_data = parser.parseNamedCustomOption("mp").AsBlob().data();
  op_data->mf_params = getDeserializedParams<nn::ImToColValid::Params>(context, memcpy_fn_data);
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* op_data = static_cast<StridedSliceOpData*>(node->user_data);

  auto memcpy = new (context->AllocatePersistentBuffer(context, sizeof(nn::ImToColValid::Params))) nn::ImToColValid(op_data->mf_params);

  //Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  //Pointers to data in In/Out Tensors
  void* in_data = const_cast<void *>(tflite::micro::GetTensorData<void>(input));
  void* out_data = tflite::micro::GetTensorData<void>(output);
  
  //memcpy->memcopy_fn((int8_t*)out_data, (int8_t*)in_data, op_data->mf_params->, op_data->begin_x, 0);
  
  return kTfLiteOk;
}

}  // namespace strided_slice_v3


TfLiteRegistration *Register_Strided_Slice_V3() {
  static TfLiteRegistration r = {strided_slice_v3::Init, nullptr, strided_slice_v3::Prepare,
                                 strided_slice_v3::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite