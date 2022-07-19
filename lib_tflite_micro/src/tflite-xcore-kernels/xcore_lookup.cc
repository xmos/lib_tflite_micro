// Copyright (c) 2022, XMOS Ltd, All rights reserved

#include "MemCpyFn.hpp"
#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace lookup {

// This is the struct that contains the data required by the operator
struct LookupOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<LookupOpData>(context);
  op_data->name = "XC_lookup";

  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<LookupOpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  int input_size = 1;
  for (int i = 0; i < input->dims->size; i++)
    input_size *= input->dims->data[i];
  const TfLiteEvalTensor *table = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  void *table_vals =
      const_cast<void *>(tflite::micro::GetTensorData<void>(table));
  void *out_data = tflite::micro::GetTensorData<void>(output);
  const int8_t *in_val = tflite::micro::GetTensorData<int8_t>(input);

  for (int i = 0; i < input_size; i++) {
    ((int8_t *)out_data)[i] = ((int8_t *)table_vals)[((uint8_t *)in_val)[i]];
  }

  return kTfLiteOk;
}

} // namespace lookup

TfLiteRegistration *Register_XC_lookup() {
  static TfLiteRegistration r = {lookup::Init, nullptr, lookup::Prepare,
                                 lookup::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
