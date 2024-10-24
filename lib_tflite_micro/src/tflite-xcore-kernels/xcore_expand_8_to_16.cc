// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"
extern "C" {
#include "lib_nn/api/nn_operator.h"
#include "lib_nn/api/expand_8_to_16.h"
}

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace expand_8to16 {

struct Expand_8_To_16Shared {
  int8_t *X;
  int16_t *Y;
};

extern "C" {
void expand_8_to_16_thread_worker(void *shared, void *start, void *count) {
  int *s = static_cast<int *>(start);
  int *c = static_cast<int *>(count);
  auto sd = static_cast<Expand_8_To_16Shared *>(shared);
  expand_8_to_16(sd->Y + *s, sd->X + *s, *c);
}
}

// This is the struct that contains the data required by the operator
struct Expand_8_To_16OpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int tc;
  int start[XCORE_MAX_NUM_THREADS];
  int count[XCORE_MAX_NUM_THREADS];
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<Expand_8_To_16OpData>(context);
  op_data->name = "XC_expand_8_to_16";
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<Expand_8_To_16OpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const TfLiteEvalTensor *input =
      tflite_micro::micro::GetEvalInput(context, node, 0);
  int input_size = tflite_micro::micro::GetTensorShape(input).FlatSize();
  op_data->tc = calculateAlignedThreadSplit(xc_config->model_thread_count, input_size, op_data->start, op_data->count);
  for (int t = 0; t < op_data->tc; t++) {
    op_data->count[t] = op_data->count[t] - op_data->start[t];
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = static_cast<Expand_8_To_16OpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input =
      tflite_micro::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  const int8_t *in_data = tflite_micro::micro::GetTensorData<int8_t>(input);
  int8_t *out_data = tflite_micro::micro::GetTensorData<int8_t>(output);

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const int tc = op_data->tc;
  Expand_8_To_16Shared shared_data;
  shared_data.X = const_cast<int8_t *>(in_data);
  shared_data.Y = (int16_t *)out_data;
  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&op_data->start[t], (void *)&op_data->count[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }
  thread_call((void *)&shared_data, &op_data->start[tc - 1], &op_data->count[tc - 1],
              (thread_function_pointer_t)expand_8_to_16_thread_worker,
              &xc_config->thread_info);
  return kTfLiteOk;
}

} // namespace expand_8to16

TFLMRegistration *Register_XC_expand_8_to_16() {
  static TFLMRegistration r = {expand_8to16::Init, nullptr, expand_8to16::Prepare,
                               expand_8to16::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
