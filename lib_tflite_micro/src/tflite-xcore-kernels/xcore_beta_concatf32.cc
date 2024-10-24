// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"
extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace beta_concatf32 {

void interleaved_memcpy(int8_t *dst, int8_t *src1, int8_t* src2, int channels_bytes, int samples_start, int samples_end) {
  for (int i = samples_start; i < samples_end; i++) {
    int src_index = i * channels_bytes;
    int dst_index = i * 2 * channels_bytes;
    memcpy(dst + dst_index, src1 + src_index, channels_bytes);
    memcpy(dst + dst_index + channels_bytes, src2 + src_index, channels_bytes);
  }
}

// This is the struct that contains the data required by the operator
struct Beta_ConcatF32OpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int tc;
  int s[XCORE_MAX_NUM_THREADS];
  int e[XCORE_MAX_NUM_THREADS];
};

struct Beta_ConcatF32Shared {
  int8_t *out;
  int8_t *in1;
  int8_t *in2;
  int channels_bytes;
};

extern "C" {
void beta_concatf32_thread_worker(void *shared, void *samples_start, void *samples_end) {
  int *s = static_cast<int *>(samples_start);
  int *e = static_cast<int *>(samples_end);
  auto sd = static_cast<Beta_ConcatF32Shared *>(shared);
  interleaved_memcpy(sd->out, sd->in1, sd->in2, sd->channels_bytes, *s, *e);
}
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<Beta_ConcatF32OpData>(context);
  op_data->name = "XC_beta_concatf32";
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<Beta_ConcatF32OpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const TfLiteEvalTensor *input1 = tflite_micro::micro::GetEvalInput(context, node, 0);
  int samples = input1->dims->data[1];
  op_data->tc = calculateAlignedThreadSplit(xc_config->model_thread_count, samples, op_data->s, op_data->e);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = static_cast<Beta_ConcatF32OpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input1 = tflite_micro::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *input2 = tflite_micro::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);

  int channels = input1->dims->data[input1->dims->size - 1];

  // Pointers to data in In/Out Tensors
  int8_t *out_data = tflite_micro::micro::GetTensorData<int8_t>(output);
  int8_t *in1_data =
      const_cast<int8_t *>(tflite_micro::micro::GetTensorData<int8_t>(input1));
  int8_t *in2_data =
      const_cast<int8_t *>(tflite_micro::micro::GetTensorData<int8_t>(input2));

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());

  // todo - this second for-loop is unpleasant
  for (int t = 0; t < op_data->tc - 1; ++t) {
    thread_variable_setup(&op_data->s[t], &op_data->e[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }

  Beta_ConcatF32Shared shared_data;
  shared_data.out = out_data;
  shared_data.in1 = in1_data;
  shared_data.in2 = in2_data;
  shared_data.channels_bytes = channels * 4;

  // Now set up shared data, shared function pointer, and data for final thread.
  thread_call((void *)&shared_data, &op_data->s[op_data->tc - 1], &op_data->e[op_data->tc - 1],
              (thread_function_pointer_t)beta_concatf32_thread_worker,
              &xc_config->thread_info);

  return kTfLiteOk;
}

} // namespace beta_concatf32

TFLMRegistration *Register_XC_beta_concatf32() {
  static TFLMRegistration r = {beta_concatf32::Init, nullptr,
                                    beta_concatf32::Prepare, beta_concatf32::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
