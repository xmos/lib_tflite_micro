// Copyright (c) 2023, XMOS Ltd, All rights reserved

extern "C" {
#include "vpu_memmove_word_aligned.h"
}

#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace broadcast {

struct BroadcastOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int32_t size;
  int32_t num_copies;
  int32_t num_broadcasts;
  void (*func_ptr)(void *, const void *, unsigned);
};

void memmove_wrapper(void *dst, const void *src, unsigned size) {
  memmove(dst, src, size);
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<BroadcastOpData>(context);
  op_data->name = "XC_broadcast";
  auto parser = CustomOptionParser(buffer, length);
  op_data->size = parser.parseNamedCustomOption("s").AsInt32();
  op_data->num_copies = parser.parseNamedCustomOption("n").AsInt32();
  op_data->num_broadcasts = parser.parseNamedCustomOption("b").AsInt32();
  bool use_vpu = parser.parseNamedCustomOption("v").AsBool();
  op_data->func_ptr = use_vpu ? vpu_memmove_word_aligned : memmove_wrapper;
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<BroadcastOpData *>(node->user_data);
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  const int8_t *in_data = tflite::micro::GetTensorData<int8_t>(input);
  int8_t *out_data = tflite::micro::GetTensorData<int8_t>(output);
  const int size = op_data->size;
  const int num_copies = op_data->num_copies;
  const int num_broadcasts = op_data->num_broadcasts;
  void (*func_ptr)(void *, const void *, unsigned) = op_data->func_ptr;
  for (int i = 0; i < num_broadcasts; i++) {
    for (int j = 0; j < num_copies; j++) {
      func_ptr(out_data, in_data, size);
      out_data += size;
    }
    in_data += size;
  }

  return kTfLiteOk;
}

} // namespace broadcast

TFLMRegistration *Register_XC_broadcast() {
  static TFLMRegistration r = {broadcast::Init, nullptr, broadcast::Prepare,
                               broadcast::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
