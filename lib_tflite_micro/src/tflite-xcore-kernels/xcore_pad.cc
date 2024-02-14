// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include <string.h>
#include "xcore_custom_options.h"
#include "xcore_utils.h"

extern "C" {
#include "nn_op_utils.h"
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace pad {

struct PadOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int32_t begin[5];
  int32_t end[5];
  int32_t in_offsets[4];
  int32_t out_offsets[4];
  int32_t zero_point;
  bool is_vpu;
};

void copy_vec(int32_t *dst, flexbuffers::Reference ref) {
  auto vec = ref.AsVector();
  for (int i = 0; i < vec.size(); i++) {
    dst[i] = vec[i].AsInt32();
  }
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<PadOpData>(context);
  op_data->name = "XC_Pad";
  auto parser = CustomOptionParser(buffer, length);
  copy_vec(op_data->begin, parser.parseNamedCustomOption("b"));
  copy_vec(op_data->end, parser.parseNamedCustomOption("e"));
  copy_vec(op_data->in_offsets, parser.parseNamedCustomOption("i"));
  copy_vec(op_data->out_offsets, parser.parseNamedCustomOption("o"));
  op_data->is_vpu = parser.parseNamedCustomOption("v").AsBool();
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  MicroContext *micro_context = GetMicroContext(context);
  auto op_data = construct_persistent_object<PadOpData>(context);
  TfLiteTensor *input = micro_context->AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  op_data->zero_point = input->params.zero_point;
  micro_context->DeallocateTempTfLiteTensor(input);
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
  // TODO [michaelp]: Make this cleaner
  // Get size of output buffer
  size_t out_size;
  TfLiteTypeSizeOf(output->type, &out_size);
  for (int i = 0; i < output->dims->size; i++) {
    out_size *= output->dims->data[i];
  }
  vpu_memset_32(out_data, op_data->zero_point, out_size / 4);
  slice_memcpy((int8_t *)out_data, (int8_t *)in_data, op_data->out_offsets,
               op_data->in_offsets, op_data->begin, op_data->end,
               op_data->is_vpu ? memcpy_wrapper : vpu_memcpy);
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
