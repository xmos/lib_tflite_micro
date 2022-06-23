// Copyright (c) 2022, XMOS Ltd, All rights reserved

#include "MemCpyFn.hpp"
#include "xcore_custom_options.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace copy_into {

// This is the struct that contains the data required by the operator
struct CopyIntoOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int32_t offset;
};

template <typename T>
T *getDeserializedParams(TfLiteContext *context, const uint8_t *data) {
  char *allocated_memory;
  int allocationByteCount = sizeof(T);
  allocated_memory =
      (char *)context->AllocatePersistentBuffer(context, allocationByteCount);
  T *param = T::template deserialise<T>(allocated_memory, (const char *)data);
  return param;
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<CopyIntoOpData>(context);
  op_data->name = "XC_Copy_Into";

  auto parser = CustomOptionParser(buffer, length);
  op_data->offset = parser.parseNamedCustomOption("offset").AsInt32();
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<CopyIntoOpData *>(node->user_data);
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *concatTensor = tflite::micro::GetEvalInput(context, node, 1);
  // Pointers to data in In/Out Tensors
  void *in_data = (void *)(tflite::micro::GetTensorData<void>(input));
  void *out_data = (void *)tflite::micro::GetTensorData<void>(concatTensor);
  // Get size of input tensor
  TfLiteIntArray* input_dims = input->dims;
  int32_t input_size = 1;
  for(int i = 0; i < input_dims->size; i++) input_size*input_dims->data[i];
  // Memcpy input to concatTensor plus offset
  memcpy((out_data + op_data->offset), in_data, input_size);

  return kTfLiteOk;
}

} // namespace copy_into

TfLiteRegistration *Register_Copy_Into() {
  static TfLiteRegistration r = {copy_into::Init, nullptr,
                                 copy_into::Prepare, copy_into::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
