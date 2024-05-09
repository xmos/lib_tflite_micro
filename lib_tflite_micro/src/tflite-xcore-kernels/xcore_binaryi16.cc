// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "xcore_common.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"
extern "C" {
#include "lib_nn/api/nn_operator.h"
#include "lib_nn/api/add_int16.h"
#include "lib_nn/api/multiply_int16.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace binaryi16 {

const BinaryI16FnType fn_ptrs[] = {(BinaryI16FnType)&add_int16_tensor,
                                   (BinaryI16FnType)&multiply_int16_tensor};

struct BinaryI16Shared {
  int16_t *X1;
  int16_t *X2;
  int16_t *Y;
  int16_t *blob;
  BinaryI16FnType fn;
};

extern "C" {
void binaryi16_thread_worker(void *shared, void *start, void *end) {
  int *s = static_cast<int *>(start);
  int *e = static_cast<int *>(end);
  auto sd = static_cast<BinaryI16Shared *>(shared);
  sd->fn(sd->Y + *s, sd->X1 + *s, sd->X2 + *s, *e - *s, sd->blob);
}
}

// This is the struct that contains the data required by the operator
struct BinaryI16OpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int tc;
  int s[XCORE_MAX_NUM_THREADS];
  int e[XCORE_MAX_NUM_THREADS];
  int opType;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<BinaryI16OpData>(context);
  op_data->name = "XC_binaryi16";

  auto parser = CustomOptionParser(buffer, length);
  op_data->opType = parser.parseNamedCustomOption("type").AsInt32();
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<BinaryI16OpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const TfLiteEvalTensor *output =
      tflite::micro::GetEvalOutput(context, node, 0);
  int output_size = tflite::micro::GetTensorShape(output).FlatSize();
  op_data->tc = xc_config->model_thread_count;
  calculateThreadSplit(op_data->tc, output_size, op_data->s, op_data->e);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = static_cast<BinaryI16OpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input1 =
      tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *input2 =
      tflite::micro::GetEvalInput(context, node, 1);
  const TfLiteEvalTensor *blob = tflite::micro::GetEvalInput(context, node, 2);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  const int16_t *in1_data = tflite::micro::GetTensorData<int16_t>(input1);
  const int16_t *in2_data = tflite::micro::GetTensorData<int16_t>(input2);
  const int16_t *blob_data = tflite::micro::GetTensorData<int16_t>(blob);
  int16_t *out_data = tflite::micro::GetTensorData<int16_t>(output);

  int output_size = tflite::micro::GetTensorShape(output).FlatSize();

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const int tc = op_data->tc;
  BinaryI16Shared shared_data;
  shared_data.Y = out_data;
  shared_data.X1 = const_cast<int16_t *>(in1_data);
  shared_data.X2 = const_cast<int16_t *>(in2_data);
  shared_data.blob = const_cast<int16_t *>(blob_data);
  shared_data.fn = fn_ptrs[op_data->opType];
  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&op_data->s[t], (void *)&op_data->e[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }
  thread_call((void *)&shared_data, &op_data->s[tc - 1], &op_data->e[tc - 1],
              (thread_function_pointer_t)binaryi16_thread_worker,
              &xc_config->thread_info);

  // shared_data.fn((void *)out_data, (void *)in1_data, (void *)in2_data,
  //                output_size, (void *)blob_data);

  return kTfLiteOk;
}

} // namespace binaryi16

TFLMRegistration *Register_XC_binaryi16() {
  static TFLMRegistration r = {binaryi16::Init, nullptr, binaryi16::Prepare,
                               binaryi16::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
