// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "tensorflow/lite/kernels/internal/reference/broadcast_to.h"

#include "xcore_custom_options.h"
#include "xcore_utils.h"
extern "C" {
#include "lib_nn/api/nn_layers.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace mul {

// This is the struct that contains the data required by the operator
struct MulOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  nn_mul_params_t *mp_params;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<MulOpData>(context);
  op_data->name = "XC_mul";

  auto parser = CustomOptionParser(buffer, length);
  op_data->mp_params = (nn_mul_params_t *)parser.parseNamedCustomOption("mp").AsBlob().data();

  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<MulOpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input1 =
      tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *input2 =
      tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  int8_t *out_data = tflite::micro::GetTensorData<int8_t>(output);
  const int8_t *in1_data = tflite::micro::GetTensorData<int8_t>(input1);
  const int8_t *in2_data = tflite::micro::GetTensorData<int8_t>(input2);

  int output_size = tflite::micro::GetTensorShape(output).FlatSize();
  int x1_size = tflite::micro::GetTensorShape(input1).FlatSize();
  int x2_size = tflite::micro::GetTensorShape(input2).FlatSize();

  if (x2_size < x1_size) {
    // Broadcast input2 into the output
    // We can then mul input1 and output to get actual output
    assert(x1_size == output_size);
    tflite::reference_ops::BroadcastTo<5>(
      tflite::micro::GetTensorShape(input2), input2->data.raw,
      tflite::micro::GetTensorShape(output), output->data.raw, input2->type);

    in2_data = out_data;
  } else if (x1_size < x2_size) {
    // Broadcast input1 into the output
    // We can then mul input2 and output to get actual output
    assert(x2_size == output_size);
    tflite::reference_ops::BroadcastTo<5>(
      tflite::micro::GetTensorShape(input1), input1->data.raw,
      tflite::micro::GetTensorShape(output), output->data.raw, input1->type);

    in1_data = out_data;
  }

  mul_elementwise(in1_data, in2_data, output_size, op_data->mp_params, out_data);

  return kTfLiteOk;
}

} // namespace mul

TFLMRegistration *Register_XC_mul() {
  static TFLMRegistration r = {mul::Init, nullptr, mul::Prepare,
                                    mul::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
