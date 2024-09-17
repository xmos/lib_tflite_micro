// Copyright (c) 2023, XMOS Ltd, All rights reserved

extern "C" {
#include "vpu_memmove_word_aligned.h"
}

#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace slice {

struct SliceOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int32_t start;
  int32_t offset;
  int32_t size;
  int32_t num_copies;
  void (*func_ptr)(void *, const void *, unsigned);
};

void memmove_wrapper(void *dst, const void *src, unsigned size) {
  memmove(dst, src, size);
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<SliceOpData>(context);
  op_data->name = "XC_Slice";
  auto parser = CustomOptionParser(buffer, length);
  op_data->start = parser.parseNamedCustomOption("s").AsInt32();
  op_data->offset = parser.parseNamedCustomOption("o").AsInt32();
  op_data->size = parser.parseNamedCustomOption("l").AsInt32();
  op_data->num_copies = parser.parseNamedCustomOption("n").AsInt32();
  bool use_vpu = parser.parseNamedCustomOption("v").AsBool();
  op_data->func_ptr = use_vpu ? vpu_memmove_word_aligned : memmove_wrapper;
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<SliceOpData *>(node->user_data);
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite_micro::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  const int8_t *in_data =
      tflite_micro::micro::GetTensorData<int8_t>(input) + op_data->start;
  int8_t *out_data = tflite_micro::micro::GetTensorData<int8_t>(output);
  const int size = op_data->size;
  const int offset = op_data->offset;
  void (*func_ptr)(void *, const void *, unsigned) = op_data->func_ptr;
  for (int i = 0; i < op_data->num_copies; i++) {
    func_ptr(out_data, in_data, size);
    in_data += offset;
    out_data += size;
  }
  return kTfLiteOk;
}

} // namespace slice

TFLMRegistration *Register_XC_slice() {
  static TFLMRegistration r = {slice::Init, nullptr, slice::Prepare,
                               slice::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
