// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "nn_op_utils.h"
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
  bool is_vpu;
};

void copy_blob(int32_t *dst, flexbuffers::Reference ref) {
  auto blob = ref.AsBlob();
  memcpy(dst, blob.data(), blob.size());
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<SliceOpData>(context);
  op_data->name = "XC_Slice";
  auto parser = CustomOptionParser(buffer, length);
  copy_blob(op_data->begin, parser.parseNamedCustomOption("b"));
  copy_blob(op_data->end, parser.parseNamedCustomOption("e"));
  copy_blob(op_data->in_offsets, parser.parseNamedCustomOption("i"));
  copy_blob(op_data->out_offsets, parser.parseNamedCustomOption("o"));
  op_data->is_vpu = parser.parseNamedCustomOption("v").AsBool();
  return op_data;
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
  for (int i0 = op_data->begin[0]; i0 < op_data->end[0]; i0++) {
    const int32_t in_idx0 = i0 * op_data->in_offsets[0];
    const int32_t out_idx0 = (i0 - op_data->begin[0]) * op_data->out_offsets[0];
    for (int i1 = op_data->begin[1]; i1 < op_data->end[1]; i1++) {
      const int32_t in_idx1 = in_idx0 + i1 * op_data->in_offsets[1];
      const int32_t out_idx1 =
          out_idx0 + (i1 - op_data->begin[1]) * op_data->out_offsets[1];
      for (int i2 = op_data->begin[2]; i2 < op_data->end[2]; i2++) {
        const int32_t in_idx2 = in_idx1 + i2 * op_data->in_offsets[2];
        const int32_t out_idx2 =
            out_idx1 + (i2 - op_data->begin[2]) * op_data->out_offsets[2];
        for (int i3 = op_data->begin[3]; i3 < op_data->end[3]; i3++) {
          const int32_t in_idx3 = in_idx2 + i3 * op_data->in_offsets[3];
          const int32_t out_idx3 =
              out_idx2 + (i3 - op_data->begin[3]) * op_data->out_offsets[3];
          if (op_data->is_vpu) {
            vpu_memcpy((int8_t *)out_data + out_idx3,
                       (int8_t *)in_data + in_idx3 + op_data->begin[4],
                       op_data->out_offsets[3]);
          } else {
            memcpy((int8_t *)out_data + out_idx3,
                   (int8_t *)in_data + in_idx3 + op_data->begin[4],
                   op_data->out_offsets[3]);
          }
        }
      }
    }
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
} // namespace tflite
