// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "xcore_custom_options.h"
#include "xcore_utils.h"
#include <string.h>

extern "C" {
#include "lib_nn/api/nn_operator.h"
#include "nn_op_utils.h"
#include "vpu_memmove_word_aligned.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace concat {

struct ConcatOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int32_t num_copies;
  int32_t size1;
  int32_t size2;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<ConcatOpData>(context);
  op_data->name = "XC_Concat";
  auto parser = CustomOptionParser(buffer, length);
  op_data->num_copies = parser.parseNamedCustomOption("n").AsInt32();
  op_data->size1 = parser.parseNamedCustomOption("s1").AsInt32();
  op_data->size2 = parser.parseNamedCustomOption("s2").AsInt32();
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  TFLITE_DCHECK(node->user_data != nullptr);

  auto *op_data = static_cast<ConcatOpData *>(node->user_data);
  const TfLiteEvalTensor *input1 =
      tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *input2 =
      tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  const int8_t *in_data1 = tflite::micro::GetTensorData<int8_t>(input1);
  const int8_t *in_data2 = tflite::micro::GetTensorData<int8_t>(input2);
  int8_t *out_data = tflite::micro::GetTensorData<int8_t>(output);
  const int size1 = op_data->size1;
  const int size2 = op_data->size2;
  for (int i = 0; i < op_data->num_copies; i++) {
    vpu_memmove_word_aligned(out_data, in_data1, size1);
    out_data += size1;
    in_data1 += size1;
    vpu_memmove_word_aligned(out_data, in_data2, size2);
    out_data += size2;
    in_data2 += size2;
  }
  return kTfLiteOk;
}

} // namespace concat

TFLMRegistration *Register_XC_concat() {
  static TFLMRegistration r = {concat::Init, nullptr, concat::Prepare,
                               concat::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
