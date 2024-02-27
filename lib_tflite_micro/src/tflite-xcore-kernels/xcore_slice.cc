// Copyright (c) 2023, XMOS Ltd, All rights reserved

extern "C" {
#include "nn_layers.h"
#include "nn_op_utils.h"
}

#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace slice {

// This is the struct that contains the data required by the operator
struct SliceOpData
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

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<SliceOpData>(context);
  op_data->name = "XC_Slice";
  auto parser = CustomOptionParser(buffer, length);
  copy_vec(op_data->begin, parser.parseNamedCustomOption("b"));
  copy_vec(op_data->end, parser.parseNamedCustomOption("e"));
  copy_vec(op_data->in_offsets, parser.parseNamedCustomOption("i"));
  copy_vec(op_data->out_offsets, parser.parseNamedCustomOption("o"));
  return op_data;
}

inline void memcpy_wrapper(void *dst, void *src, size_t size) {
  memcpy(dst, src, size);
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

// [michael p] TFLite supports up to 5D tensors, last loop is replaced with
// memcpy. If there are less than 5 dimensions: the first 5 - D elements of
// begin/end are 0 and 1 respectively. To ensure optimal performance, the
// compiler will re-write begin/end/offsets to have the sliced dimension fused
// with the dimensions after it. Example: input_size = [10, 20, 5], begin = [1,
// 0, 0], end = [8, 20, 5] will give begin = [0, 0, 0, 0, 1*20*5], end = [1, 1,
// 1, 1, 8*20*5] as if shape was [1, 1, 1, 1, 10*20*5], allowing us to use
// vpu_memcpy since the last dimension is a multiple of 4.
//
// If slice becomes a bottleneck:
//
// - For less memory usage, you can get rid of a few for loops, the operator
// supports slicing along the number of for loops axes. 5 for loops are only
// necessary if you slice along each axis of a batched video input.
//
// - For better performance without the last dimension being a multiple of 4,
// you can check if vpu_memcpy_vec can be used on some iterations (up to 2x
// improvement)
TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<SliceOpData *>(node->user_data);
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  const void *in_data = tflite::micro::GetTensorData<void>(input);
  void *out_data = tflite::micro::GetTensorData<void>(output);
  slice_memcpy((int8_t *)out_data, (int8_t *)in_data, op_data->in_offsets,
               op_data->out_offsets, op_data->begin, op_data->end,
               memcpy_wrapper);
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
} // namespace tflite
