// Copyright (c) 2022, XMOS Ltd, All rights reserved

#include "MemCpyFn.hpp"
#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace connector {

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

} // namespace connector

TfLiteRegistration *Register_XC_connector() {
  static TfLiteRegistration r = {connector::Init, nullptr,
                                 connector::Prepare, connector::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
