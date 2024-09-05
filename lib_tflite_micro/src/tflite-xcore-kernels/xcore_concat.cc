// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "xcore_custom_options.h"
#include "xcore_utils.h"
#include <string.h>

extern "C" {
#include "vpu_memmove_word_aligned.h"
}

constexpr int kMaxNumInputs = 13;

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace concat {

using tflite::micro::GetEvalInput;
using tflite::micro::GetEvalOutput;
using tflite::micro::GetTensorData;

struct ConcatOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int32_t num_copies;
  int32_t sizes[kMaxNumInputs];
  int32_t num_inputs;
  void (*func_ptr)(void *, const void *, unsigned);
};

void memmove_wrapper(void *dst, const void *src, unsigned size) {
  memmove(dst, src, size);
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<ConcatOpData>(context);
  op_data->name = "XC_Concat";
  auto parser = CustomOptionParser(buffer, length);
  op_data->num_copies = parser.parseNamedCustomOption("n").AsInt32();
  op_data->num_inputs = parser.parseNamedCustomOption("i").AsInt32();
  auto sizes = parser.parseNamedCustomOption("s").AsVector();
  TFLITE_DCHECK(sizes.size() <= kMaxNumInputs);
  for (int i = 0; i < sizes.size(); i++) {
    op_data->sizes[i] = sizes[i].AsInt32();
  }
  bool use_vpu = parser.parseNamedCustomOption("v").AsBool();
  op_data->func_ptr = use_vpu ? vpu_memmove_word_aligned : memmove_wrapper;
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  TFLITE_DCHECK(node->user_data != nullptr);

  auto *op_data = static_cast<ConcatOpData *>(node->user_data);
  const int8_t *inputs[kMaxNumInputs];
  for (int i = 0; i < op_data->num_inputs; i++)
    inputs[i] = GetTensorData<int8_t>(GetEvalInput(context, node, i));

  TfLiteEvalTensor *output = GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  int8_t *out_data = GetTensorData<int8_t>(output);
  void (*func_ptr)(void *, const void *, unsigned) = op_data->func_ptr;
  bool all_ones = true;
  for (int i = 0; i < op_data->num_inputs; i++) {
    if (op_data->sizes[i] != 1) {
      all_ones = false;
      break;
    }
  }
  if (all_ones) {
    for (int i = 0; i < op_data->num_copies; i++) {
      for (int j = 0; j < op_data->num_inputs; j++) {
        *out_data++ = *inputs[j]++;
        // out_data[0] = inputs[j][0];
        // out_data++;
        // inputs[j]++;
      }
    }
    return kTfLiteOk;
  }
  for (int i = 0; i < op_data->num_copies; i++) {
    for (int j = 0; j < op_data->num_inputs; j++) {
      const int size = op_data->sizes[j];
      func_ptr(out_data, inputs[j], size);
      out_data += size;
      inputs[j] += size;
    }
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
