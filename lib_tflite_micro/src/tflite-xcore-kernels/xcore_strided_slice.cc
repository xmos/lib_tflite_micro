// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "MemCpyFn.hpp"
#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace strided_slice {

enum MemcpyType {
  SliceCpy_t,
  VpuCpy_t,
  MemCpy_t,
};

// This is the struct that contains the data required by the operator
struct StridedSliceOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int32_t begin_x;
  int32_t begin_y;
  nn::ImToColValid::Params *mf_params;
  nn::ImToColValid *memcpy;
  int32_t memcpy_type;
};

template <typename T>
T *getDeserializedParams(TfLiteContext *context, const uint8_t *data) {
  char *allocated_memory;
  int allocationByteCount = sizeof(T);
  allocated_memory =
      (char *)context->AllocatePersistentBuffer(context, allocationByteCount);
  T *param = T::template deserialise<T>(allocated_memory, (const char *)data);
  return param;
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<StridedSliceOpData>(context);
  op_data->name = "XC_Strided_Slice";

  auto parser = CustomOptionParser(buffer, length);
  const uint8_t *memcpy_fn_data =
      parser.parseNamedCustomOption("mp").AsBlob().data();
  op_data->mf_params =
      getDeserializedParams<nn::ImToColValid::Params>(context, memcpy_fn_data);
  op_data->begin_x = parser.parseNamedCustomOption("begin_x").AsInt32();
  op_data->begin_y = parser.parseNamedCustomOption("begin_y").AsInt32();
  op_data->memcpy = new (context->AllocatePersistentBuffer(
      context, sizeof(nn::ImToColValid::Params)))
      nn::ImToColValid(op_data->mf_params);
  op_data->memcpy_type = parser.parseNamedCustomOption("type").AsInt32();
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<StridedSliceOpData *>(node->user_data);
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  void *in_data = const_cast<void *>(tflite::micro::GetTensorData<void>(input));
  void *out_data = tflite::micro::GetTensorData<void>(output);

  if (op_data->memcpy_type == SliceCpy_t) {
    size_t input_offset = Offset(tflite::micro::GetTensorShape(input), 0,
                                 op_data->begin_y, op_data->begin_x, 0);
    size_t num_bytes = (op_data->mf_params->input_height + 1) *
                       op_data->mf_params->bytes_per_h_line;
    memcpy((int8_t *)out_data, (int8_t *)in_data + input_offset, num_bytes);
  } else if (op_data->memcpy_type == VpuCpy_t) {
    op_data->memcpy->memcopy_fn((int8_t *)out_data, (int8_t *)in_data,
                                op_data->begin_y, op_data->begin_x, 0);
  } else if (op_data->memcpy_type == MemCpy_t) {
    int num_in_bytes = input->dims->data[3];
    int num_out_bytes = output->dims->data[3];
    int num_of_pixels = output->dims->data[1] * output->dims->data[2];
    for (int i = 0; i < num_of_pixels; i++) {
      memcpy((int8_t *)out_data + (i * num_out_bytes),
             (int8_t *)in_data + (i * num_in_bytes), num_out_bytes);
    }
  } else {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

} // namespace strided_slice

TfLiteRegistration_V1 *Register_XC_strided_slice() {
  static TfLiteRegistration_V1 r = {strided_slice::Init, nullptr,
                                    strided_slice::Prepare,
                                    strided_slice::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
