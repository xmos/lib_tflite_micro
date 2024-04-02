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
namespace padv2 {

struct PadOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int32_t begin[5];
  int32_t end[5];
  int32_t in_offsets[4];
  int32_t out_offsets[4];
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

inline void inv_vpu_memcpy_wrapper(void *src, void *dst, size_t len) {
  vpu_memmove_word_aligned(dst, src, len);
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<PadOpData>(context);
  op_data->name = "XC_PadV2";
  auto parser = CustomOptionParser(buffer, length);
  copy_vec(op_data->begin, parser.parseNamedCustomOption("b"));
  copy_vec(op_data->end, parser.parseNamedCustomOption("e"));
  copy_vec(op_data->in_offsets, parser.parseNamedCustomOption("i"));
  copy_vec(op_data->out_offsets, parser.parseNamedCustomOption("o"));
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  TFLITE_DCHECK(node->user_data != nullptr);

  auto *op_data = static_cast<PadOpData *>(node->user_data);
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  const void *in_data = tflite::micro::GetTensorData<void>(input);
  void *out_data = tflite::micro::GetTensorData<void>(output);
  // TODO [michael p]: Make this cleaner
  // Get size of output buffer
  size_t out_size;
  TfLiteTypeSizeOf(output->type, &out_size);
  for (int i = 0; i < output->dims->size; i++) {
    out_size *= output->dims->data[i];
  }
  vpu_memset_32(out_data, 0, out_size / 4);
  slice_memcpy(
      (int8_t *)in_data, (int8_t *)out_data, op_data->in_offsets,
      op_data->out_offsets, op_data->begin, op_data->end,
      (op_data->in_offsets[3] % 4 == 0 && op_data->out_offsets[3] % 4 == 0)
          ? inv_vpu_memcpy_wrapper
          : inv_memcpy_wrapper);
  return kTfLiteOk;
}

} // namespace padv2

TFLMRegistration *Register_XC_pad_v2() {
  static TFLMRegistration r = {padv2::Init, nullptr, padv2::Prepare,
                               padv2::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
