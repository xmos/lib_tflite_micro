// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include <cstdint>
extern "C" {
#include "vpu_memmove_word_aligned.h"
#include "vpu_memset_256.h"
}

#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite_micro {
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
  const TfLiteEvalTensor *input = tflite_micro::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  const int8_t *in_data = tflite_micro::micro::GetTensorData<int8_t>(input);
  int8_t *out_data = tflite_micro::micro::GetTensorData<int8_t>(output);
  const int size = op_data->size;
  const int num_copies = op_data->num_copies;
  const int num_broadcasts = op_data->num_broadcasts;
  if (size == 1 && num_copies < 64) {
    for (int i = 0; i < num_broadcasts; i++) {
      memset(out_data, *in_data, num_copies);
      out_data += num_copies;
      in_data++;
    }
    return kTfLiteOk;
  }
  if ((size != 1 && size != 2 && size != 4) || num_copies < 64) {
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
  uint32_t c;
  uint8_t from[32];
  for (int i = 0; i < num_broadcasts; i++) {
    switch (size) {
    case 1:
      // c = ins[0] * 0x01010101;
      c = ((uint8_t)(*in_data)) * 0x01010101;
      break;
    case 2:
      c = ((uint8_t)(*in_data) | ((uint8_t)(in_data[1]) << 8)) * 0x00010001;
      break;
    case 4:
      c = ((uint8_t)(*in_data) | ((uint8_t)(in_data[1]) << 8) |
           ((uint8_t)(in_data[2]) << 16) | ((uint8_t)(in_data[3]) << 24));
      break;
    }
    broadcast_32_to_256(from, c);
    vpu_memset_256(out_data, from, num_copies * size);
    out_data += num_copies * size;
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
} // namespace tflite_micro
