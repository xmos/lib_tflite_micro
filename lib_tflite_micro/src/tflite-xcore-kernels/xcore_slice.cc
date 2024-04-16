// Copyright (c) 2023, XMOS Ltd, All rights reserved

extern "C" {
#include "nn_layers.h"
#include "nn_op_utils.h"
#include "vpu_memmove_word_aligned.h"
}

#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace slice {

struct SliceOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int32_t start;
  int32_t offset;
  int32_t size;
  int32_t num_copies;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<SliceOpData>(context);
  op_data->name = "XC_Slice";
  auto parser = CustomOptionParser(buffer, length);
  op_data->start = parser.parseNamedCustomOption("s").AsInt32();
  op_data->offset = parser.parseNamedCustomOption("o").AsInt32();
  op_data->size = parser.parseNamedCustomOption("l").AsInt32();
  op_data->num_copies = parser.parseNamedCustomOption("n").AsInt32();
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<SliceOpData *>(node->user_data);
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  const int8_t *in_data =
      tflite::micro::GetTensorData<int8_t>(input) + op_data->start;
  int8_t *out_data = tflite::micro::GetTensorData<int8_t>(output);
  const int size = op_data->size;
  const int offset = op_data->offset;
  for (int i = 0; i < op_data->num_copies; i++) {
    vpu_memmove_word_aligned(out_data, in_data, size);
    in_data += offset;
    out_data += size;
  }
  return kTfLiteOk;
}

} // namespace slice

TFLMRegistration *Register_XC_slice() {
  static TFLMRegistration r = {slice::Init, nullptr, slice::Prepare,
                               slice::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
