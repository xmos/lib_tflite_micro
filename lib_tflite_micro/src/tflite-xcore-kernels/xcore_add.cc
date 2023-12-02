// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "tensorflow/lite/kernels/internal/reference/broadcast_to.h"

#include "xcore_custom_options.h"
#include "xcore_utils.h"
extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace add {

// This is the struct that contains the data required by the operator
struct AddOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  nn_add_params_t params;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<AddOpData>(context);
  op_data->name = "XC_Add";

  auto parser = CustomOptionParser(buffer, length);
  int m1 = parser.parseNamedCustomOption("m1").AsInt32();
  int m2 = parser.parseNamedCustomOption("m2").AsInt32();
  int bias = parser.parseNamedCustomOption("bias").AsInt32();
  int shift = parser.parseNamedCustomOption("shift").AsInt32();

  // Broadcast values into vectors
  // We are VLMACC-ing in 16-bit mode
  for (int i = 0; i < VPU_INT16_VLMACC_ELMS; i++) {
    op_data->params.m1[i] = (int16_t)m1;
    op_data->params.m2[i] = (int16_t)m2;
    op_data->params.shift[i] = (int16_t)shift;
    // Split 32-bit bias into two 16-bit values
    op_data->params.bias_hi[i] = bias >> 16;
    op_data->params.bias_lo[i] = (int16_t) (bias & 0XFFFF);
  }

  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<AddOpData *>(node->user_data);
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input1 =
      tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *input2 =
      tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  int8_t *in1_data =
      const_cast<int8_t *>(tflite::micro::GetTensorData<int8_t>(input1));
  int8_t *in2_data =
      const_cast<int8_t *>(tflite::micro::GetTensorData<int8_t>(input2));
  int8_t *out_data = tflite::micro::GetTensorData<int8_t>(output);

  int output_size = tflite::micro::GetTensorShape(output).FlatSize();
  int x1_size = tflite::micro::GetTensorShape(input1).FlatSize();
  int x2_size = tflite::micro::GetTensorShape(input2).FlatSize();

  if (x2_size < x1_size) {
    // Broadcast input2 into the output
    // We can then add input1 and output to get actual output
    assert(x1_size == output_size);
    tflite::reference_ops::BroadcastTo<5>(
      tflite::micro::GetTensorShape(input2), input2->data.raw,
      tflite::micro::GetTensorShape(output), output->data.raw, input2->type);

    in2_data = out_data;
  } else if (x1_size < x2_size) {
    // Broadcast input1 into the output
    // We can then add input2 and output to get actual output
    assert(x2_size == output_size);
    tflite::reference_ops::BroadcastTo<5>(
      tflite::micro::GetTensorShape(input1), input1->data.raw,
      tflite::micro::GetTensorShape(output), output->data.raw, input1->type);

    in1_data = out_data;
  }
  add_elementwise(out_data, in1_data, in2_data, &op_data->params, 0, output_size);

  return kTfLiteOk;
}

} // namespace add

TFLMRegistration *Register_XC_add() {
  static TFLMRegistration r = {add::Init, nullptr, add::Prepare, add::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
