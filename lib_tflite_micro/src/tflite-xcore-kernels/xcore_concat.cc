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
struct PadOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int32_t begin1[5];
  int32_t end1[5];
  int32_t in_offsets1[4];
  int32_t out_offsets1[4];
  int32_t begin2[5];
  int32_t end2[5];
  int32_t in_offsets2[4];
  int32_t out_offsets2[4];
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
  auto op_data = construct_persistent_object<PadOpData>(context);
  op_data->name = "XC_Concat";
  auto parser = CustomOptionParser(buffer, length);
  copy_vec(op_data->begin1, parser.parseNamedCustomOption("b1"));
  copy_vec(op_data->end1, parser.parseNamedCustomOption("e1"));
  copy_vec(op_data->in_offsets1, parser.parseNamedCustomOption("i1"));
  copy_vec(op_data->out_offsets1, parser.parseNamedCustomOption("o1"));
  copy_vec(op_data->begin2, parser.parseNamedCustomOption("b2"));
  copy_vec(op_data->end2, parser.parseNamedCustomOption("e2"));
  copy_vec(op_data->in_offsets2, parser.parseNamedCustomOption("i2"));
  copy_vec(op_data->out_offsets2, parser.parseNamedCustomOption("o2"));
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  TFLITE_DCHECK(node->user_data != nullptr);

  auto *op_data = static_cast<PadOpData *>(node->user_data);
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
  slice_memcpy((int8_t *)in_data1, (int8_t *)out_data, op_data->in_offsets1,
               op_data->out_offsets1, op_data->begin1, op_data->end1,
               inv_memcpy_wrapper);
  slice_memcpy((int8_t *)in_data2, (int8_t *)out_data, op_data->in_offsets2,
               op_data->out_offsets2, op_data->begin2, op_data->end2,
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
