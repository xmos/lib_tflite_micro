// Copyright (c) 2022, XMOS Ltd, All rights reserved

#include "MemCpyFn.hpp"
#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace pass_thru {

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  void *in_data = (void *)(tflite::micro::GetTensorData<void>(input));
  void *out_data = (void *)tflite::micro::GetTensorData<void>(output);
  // Get size of input tensor
  TfLiteIntArray* input_dims = input->dims;
  int32_t input_size = 1;
  for(int i = 0; i < input_dims->size; i++) input_size*input_dims->data[i];
  // Memcpy input to concatTensor plus offset
  memcpy(out_data, in_data, input_size);
  return kTfLiteOk;
}

} // namespace pass_thru

TfLiteRegistration *Register_XC_pass_thru() {
  static TfLiteRegistration r = {pass_thru::Init, nullptr,
                                 pass_thru::Prepare, pass_thru::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
