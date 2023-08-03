// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "tensorflow/lite/kernels/internal/reference/broadcast_to.h"

#include "MemCpyFn.hpp"
#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace mul {

// This is the struct that contains the data required by the operator
struct MulOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int B;
  int S;
  int lhsZeroPoint;
  int rhsZeroPoint;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<MulOpData>(context);
  op_data->name = "XC_mul";

  auto parser = CustomOptionParser(buffer, length);
  op_data->B = parser.parseNamedCustomOption("B").AsInt32();
  op_data->S = parser.parseNamedCustomOption("S").AsInt32();

  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  MicroContext *micro_context = GetMicroContext(context);
  TfLiteTensor *input1 = micro_context->AllocateTempInputTensor(node, 0);
  TfLiteTensor *input2 = micro_context->AllocateTempInputTensor(node, 1);

  MulOpData *data = static_cast<MulOpData *>(node->user_data);
  // Negating the zero points here, so that we don't have to use subtract in the
  // final equation
  // (((x0*x1) + (x1*-b0) + (x0*-b1)) * S + (1<<13) >> 14 + B) + (1<<5) >> 6
  data->lhsZeroPoint = -input1->params.zero_point;
  data->rhsZeroPoint = -input2->params.zero_point;

  micro_context->DeallocateTempTfLiteTensor(input1);
  micro_context->DeallocateTempTfLiteTensor(input2);
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

  for (int i = 0; i < output_size; i++) {
    int result =
        (((((op_data->S * (op_data->rhsZeroPoint * in1_data[i] +
                           op_data->lhsZeroPoint * in2_data[i] +
                           in1_data[i] * in2_data[i])) +
            (1 << 13)) >>
           14) +
          op_data->B) +
         (1 << 5)) >>
        6;
    out_data[i] = std::min(std::max(result, -128), 127);
  }

  return kTfLiteOk;
}

} // namespace mul

TfLiteRegistration_V1 *Register_XC_mul() {
  static TfLiteRegistration_V1 r = {mul::Init, nullptr, mul::Prepare,
                                    mul::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
