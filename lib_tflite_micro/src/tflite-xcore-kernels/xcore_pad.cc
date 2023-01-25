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
  nn_pad_plan_t pad_plan;
  uint32_t pad_value;
  // uint32_t n_threads
};


void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto op_data = construct_persistent_object<OpData>(context);

  auto parser = CustomOptionParser(buffer, length);
  auto plan = parser.parseNamedCustomOption("pp").AsBlob().data();
  auto pad_value = parser.parseNamedCustomOption("pv").AsUInt32();
  memcpy(&(op_data->pad_plan), plan, sizeof (nn_pad_plan_t));
  op_data->pad_value = pad_value;
  // op_data->n_threads = parser.parseNamedCustomOption("n_threads").AsUInt32();
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

  pad_run((char*)output_p,
          (char*)input_p,
          &data->pad_plan, data->pad_value);

  return kTfLiteOk;
}

}  // namespace pad

TfLiteRegistration *Register_XC_pad() {
  static TfLiteRegistration r = {pad::Init, nullptr, pad::Prepare, pad::Eval};
  return &r;
}

} // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
