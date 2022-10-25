// Copyright (c) 2022, XMOS Ltd, All rights reserved

#include "MemCpyFn.hpp"
#include "xcore_custom_options.h"
#include "xcore_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

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
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* output = GetOutput(context, node, 0);
  memcpy(output->data.raw, input->data.raw, input->bytes);
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
