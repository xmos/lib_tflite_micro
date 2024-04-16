// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "xcore_custom_options.h"
#include "xcore_utils.h"
#include <string.h>

extern "C" {
#include "lib_nn/api/nn_operator.h"
#include "vpu_memmove_word_aligned.h"
#include "vpu_memset_256.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace pad {

struct PadOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int32_t start;
  int32_t pad_size;
  int32_t size;
  int32_t num_copies;
  int32_t zero_point;
  int32_t end;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<PadOpData>(context);
  op_data->name = "XC_Pad";
  auto parser = CustomOptionParser(buffer, length);
  op_data->start = parser.parseNamedCustomOption("s").AsInt32();
  op_data->pad_size = parser.parseNamedCustomOption("p").AsInt32();
  op_data->size = parser.parseNamedCustomOption("l").AsInt32();
  op_data->num_copies = parser.parseNamedCustomOption("n").AsInt32();
  op_data->zero_point = parser.parseNamedCustomOption("z").AsInt32();
  op_data->end = parser.parseNamedCustomOption("e").AsInt32();
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  TFLITE_DCHECK(node->user_data != nullptr);

  auto *op_data = static_cast<PadOpData *>(node->user_data);
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  const int8_t *in_data = tflite::micro::GetTensorData<int8_t>(input);
  int8_t *out_data = tflite::micro::GetTensorData<int8_t>(output);
  uint8_t from[32];
  broadcast_32_to_256(from, op_data->zero_point);
  vpu_memset_256(out_data, from, op_data->start);
  out_data += op_data->start;
  const int size = op_data->size;
  const int pad_size = op_data->pad_size;
  for (int i = 0; i < op_data->num_copies; i++) {
    vpu_memmove_word_aligned(out_data, in_data, size);
    out_data += size;
    in_data += size;
    vpu_memset_256(out_data, from, pad_size);
    out_data += pad_size;
  }
  vpu_memmove_word_aligned(out_data, in_data, size);
  out_data += size;
  vpu_memset_256(out_data, from, op_data->end);
  return kTfLiteOk;
}

} // namespace pad

TFLMRegistration *Register_XC_pad() {
  static TFLMRegistration r = {pad::Init, nullptr, pad::Prepare, pad::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
