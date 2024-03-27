// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "xcore_custom_options.h"
#include "xcore_utils.h"
#include <string.h>

extern "C" {
#include "lib_nn/api/nn_operator.h"
#include "nn_op_utils.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace concat {

// TODO: [michael p] Optimise this, don't need all those params
struct ConcatOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int32_t num_copies;
  int32_t size1;
  int32_t size2;
};

inline void memcpy_wrapper(void *dst, void *src, size_t size) {
  memcpy(dst, src, size);
}

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
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input1 =
      tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *input2 =
      tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  const void *in_data1 = tflite::micro::GetTensorData<void>(input1);
  const void *in_data2 = tflite::micro::GetTensorData<void>(input2);
  void *out_data = tflite::micro::GetTensorData<void>(output);
  const int32_t offset = op_data->size1 + op_data->size2;
  slice_memcpy_1d((int8_t *)out_data, (int8_t *)in_data1, op_data->size1,
                  offset, op_data->num_copies, memcpy_wrapper);
  slice_memcpy_1d((int8_t *)out_data + op_data->size1, (int8_t *)in_data2,
                  op_data->size2, offset, op_data->num_copies, memcpy_wrapper);
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
