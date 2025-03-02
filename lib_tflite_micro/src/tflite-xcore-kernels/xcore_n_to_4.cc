// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "xcore_custom_options.h"
#include "xcore_utils.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace pad_n_to_4 {

struct OpData {
  uint32_t n;
  uint32_t pad_val;
};


void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto op_data = construct_persistent_object<OpData>(context);

  auto parser = CustomOptionParser(buffer, length);
  auto pad_value = parser.parseNamedCustomOption("pv").AsUInt32();
  op_data->pad_val = pad_value;
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);
  auto shape = tflite_micro::micro::GetTensorShape(output);
  TFLITE_DCHECK(shape.DimensionsCount() == 4 && shape.DimsData()[0] == 1);
  int number_of_pixels = shape.DimsData()[1] * shape.DimsData()[2];
  OpData* op_data = static_cast<OpData*>(node->user_data);
  op_data->n = number_of_pixels;
  return kTfLiteOk;
}

TfLiteStatus Eval3To4(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite_micro::micro::GetEvalInput(context, node, /*index=*/0);

  TfLiteEvalTensor* output =
      tflite_micro::micro::GetEvalOutput(context, node, /*index=*/0);

  int8_t *output_p =
      const_cast<int8_t *>(tflite_micro::micro::GetTensorData<int8_t>(output));
  int8_t *input_p =
      const_cast<int8_t *>(tflite_micro::micro::GetTensorData<int8_t>(input));

  // The function takes the number of pixels as data->n
  pad_3_to_4_run(output_p,
          input_p,
          data->n, data->pad_val);

  return kTfLiteOk;
}

TfLiteStatus Eval1To4(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite_micro::micro::GetEvalInput(context, node, /*index=*/0);

  TfLiteEvalTensor* output =
      tflite_micro::micro::GetEvalOutput(context, node, /*index=*/0);

  int8_t *output_p =
      const_cast<int8_t *>(tflite_micro::micro::GetTensorData<int8_t>(output));
  int8_t *input_p =
      const_cast<int8_t *>(tflite_micro::micro::GetTensorData<int8_t>(input));

  // The function takes the number of 4 byte chunks, hence aligned number of pixels/4
  int n = (data->n + 3) / 4;
  pad_1_to_4_run(output_p,
          input_p,
          n, data->pad_val);

  return kTfLiteOk;
}

}  // namespace pad

TFLMRegistration *Register_XC_pad_3_to_4() {
  static TFLMRegistration r = {pad_n_to_4::Init, nullptr, pad_n_to_4::Prepare, pad_n_to_4::Eval3To4};
  return &r;
}

TFLMRegistration *Register_XC_pad_1_to_4() {
  static TFLMRegistration r = {pad_n_to_4::Init, nullptr, pad_n_to_4::Prepare, pad_n_to_4::Eval1To4};
  return &r;
}

} // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite_micro
