// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include <string.h>
#include "xcore_custom_options.h"
#include "xcore_utils.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace pad {

struct OpData {
  uint32_t n_3;
  uint32_t pad_val;
};


void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto op_data = construct_persistent_object<OpData>(context);

  auto parser = CustomOptionParser(buffer, length);
  auto n_3 = parser.parseNamedCustomOption("n3").AsUInt32();
  auto pad_value = parser.parseNamedCustomOption("pv").AsUInt32();
  op_data->n_3 = n_3;
  op_data->pad_value = pad_value;
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, /*index=*/0);

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, /*index=*/0);

  int8_t *output_p =
      const_cast<int8_t *>(tflite::micro::GetTensorData<int8_t>(output));
  int8_t *input_p =
      const_cast<int8_t *>(tflite::micro::GetTensorData<int8_t>(input));

  pad_3_to_4_run((char*)output_p,
          (char*)input_p,
          data->n_3, data->pad_val);

  return kTfLiteOk;
}

}  // namespace pad

TfLiteRegistration *Register_XC_pad_3_to_4() {
  static TfLiteRegistration r = {pad::Init, nullptr, pad::Prepare, pad::Eval};
  return &r;
}

} // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
