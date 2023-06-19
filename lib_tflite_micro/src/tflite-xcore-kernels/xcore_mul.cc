// Copyright (c) 2023, XMOS Ltd, All rights reserved

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
  int input1_size = 1;
  for (int i = 0; i < input1->dims->size; i++)
    input1_size *= input1->dims->data[i];
  const TfLiteEvalTensor *input2 =
      tflite::micro::GetEvalInput(context, node, 1);
  int input2_size = 1;
  for (int i = 0; i < input2->dims->size; i++)
    input2_size *= input2->dims->data[i];
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  int8_t *out_data = tflite::micro::GetTensorData<int8_t>(output);
  const int8_t *in1_val = tflite::micro::GetTensorData<int8_t>(input1);
  const int8_t *in2_val = tflite::micro::GetTensorData<int8_t>(input2);

  int in2_index = 0;
  for (int i = 0; i < input1_size; i++) {
    if (in2_index == input2_size) {
      in2_index = 0;
    }
    out_data[i] =
        (((((op_data->S * (op_data->rhsZeroPoint * in1_val[i] +
                           op_data->lhsZeroPoint * in2_val[in2_index] +
                           in1_val[i] * in2_val[in2_index])) +
            (1 << 13)) >>
           14) +
          op_data->B) +
         (1 << 5)) >>
        6;
    in2_index++;
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
