// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "xcore_custom_options.h"
#include "xcore_utils.h"
#include <iostream>
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
  int32_t begin;
  int32_t in_offsets[2];
  int32_t out_offsets1[2];
  int32_t out_offsets2[2];
  int32_t end1[2];
  int32_t end2[2];
};

void copy_vec(int32_t *dst, flexbuffers::Reference ref) {
  auto vec = ref.AsVector();
  for (int i = 0; i < vec.size(); i++) {
    dst[i] = vec[i].AsInt32();
  }
}

inline void inv_memcpy_wrapper(void *src, void *dst, size_t len) {
  memcpy(dst, src, len);
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<ConcatOpData>(context);
  op_data->name = "XC_Concat";
  auto parser = CustomOptionParser(buffer, length);
  op_data->begin = parser.parseNamedCustomOption("b").AsInt32();
  copy_vec(op_data->in_offsets, parser.parseNamedCustomOption("i"));
  copy_vec(op_data->out_offsets1, parser.parseNamedCustomOption("o1"));
  copy_vec(op_data->out_offsets2, parser.parseNamedCustomOption("o2"));
  copy_vec(op_data->end1, parser.parseNamedCustomOption("e1"));
  copy_vec(op_data->end2, parser.parseNamedCustomOption("e2"));
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
  // TODO [michael p]: Make this cleaner
  // Get size of output buffer
  const int32_t io0 = op_data->in_offsets[0];
  const int32_t oo01 = op_data->out_offsets1[0];
  const int32_t oo02 = op_data->out_offsets2[0];
  const int32_t in_offsets[4] = {io0, io0, io0, op_data->in_offsets[1]};
  const int32_t out_offsets1[4] = {oo01, oo01, oo01, op_data->out_offsets1[1]};
  const int32_t out_offsets2[4] = {oo02, oo02, oo02, op_data->out_offsets2[1]};
  const int32_t end1[5] = {1, 1, 1, op_data->end1[0], op_data->end1[1]};
  const int32_t end2[5] = {1, 1, 1, op_data->end2[0], op_data->end2[1]};
  slice_memcpy((int8_t *)in_data1, (int8_t *)out_data, in_offsets, out_offsets1,
               (const int32_t[]){0, 0, 0, 0, 0}, end1, inv_memcpy_wrapper);
  slice_memcpy((int8_t *)in_data2, (int8_t *)out_data, in_offsets, out_offsets2,
               (const int32_t[]){0, 0, 0, 0, op_data->begin}, end2,
               inv_memcpy_wrapper);
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
