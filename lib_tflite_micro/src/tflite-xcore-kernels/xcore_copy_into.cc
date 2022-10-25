// Copyright (c) 2022, XMOS Ltd, All rights reserved

#include "MemCpyFn.hpp"
#include "xcore_custom_options.h"
#include "xcore_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

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
  // Get Input Tensors
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* concatTensor = GetInput(context, node, 1);

  auto *op_data = static_cast<CopyIntoOpData *>(node->user_data);

  // Memcpy input to concatTensor plus offset
  memcpy(concatTensor->data.raw + op_data->offset, input->data.raw, input->bytes);

  return kTfLiteOk;
}

} // namespace copy_into

TfLiteRegistration *Register_XC_copy_into() {
  static TfLiteRegistration r = {copy_into::Init, nullptr,
                                 copy_into::Prepare, copy_into::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
