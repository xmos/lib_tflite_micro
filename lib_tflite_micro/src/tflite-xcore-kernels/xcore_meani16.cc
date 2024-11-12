
// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "xcore_custom_options.h"
#include "xcore_utils.h"
extern "C" {
#include "lib_nn/api/nn_layers.h"
}

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace meani16 {

// This is the struct that contains the data required by the operator
struct MeanI16OpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int start_dim_size;
  int mean_dim_size;
  int end_dim_size;
  float scale_mul;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<MeanI16OpData>(context);
  op_data->name = "XC_meani16";

  auto parser = CustomOptionParser(buffer, length);
  op_data->start_dim_size = parser.parseNamedCustomOption("s").AsInt32();
  op_data->mean_dim_size = parser.parseNamedCustomOption("m").AsInt32();
  op_data->end_dim_size = parser.parseNamedCustomOption("e").AsInt32();
  op_data->scale_mul = parser.parseNamedCustomOption("sm").AsFloat();
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<MeanI16OpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input =
      tflite_micro::micro::GetEvalInput(context, node, 0);

  TfLiteEvalTensor *output =
      tflite_micro::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  int16_t *out_data = tflite_micro::micro::GetTensorData<int16_t>(output);
  const int16_t *in_data = tflite_micro::micro::GetTensorData<int16_t>(input);
  mean_int16(in_data, out_data, op_data->start_dim_size, op_data->mean_dim_size,
             op_data->end_dim_size, op_data->scale_mul);

  return kTfLiteOk;
}

} // namespace meani16

TFLMRegistration *Register_XC_meani16() {
  static TFLMRegistration r = {meani16::Init, nullptr, meani16::Prepare,
                               meani16::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
