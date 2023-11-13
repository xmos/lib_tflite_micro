// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "conv2d_float.h"
#include "xcore_common.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace beta_convf32 {

// This is the struct that contains the data required by the operator
struct Beta_ConvF32OpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int tc;
  int s[XCORE_MAX_NUM_THREADS];
  int e[XCORE_MAX_NUM_THREADS];
};

struct Beta_ConvF32Shared {
  float *out;
  float *in;
  float *kernels;
  float *biases;
  int out_w, out_h, out_d;
  int in_w, in_h, in_d;
};

extern "C" {
void beta_convf32_thread_worker(void *shared, void *d_start, void *d_end) {
  int *s = static_cast<int *>(d_start);
  int *e = static_cast<int *>(d_end);
  auto sd = static_cast<Beta_ConvF32Shared *>(shared);
#ifdef NN_USE_REF
  xc_conv2d_float_kw5xh2_stride_w3_ref(sd->out, sd->in, sd->kernels, sd->biases,
                                       sd->out_w, sd->out_h, sd->out_d,
                                       sd->in_w, sd->in_h, sd->in_d);
#else
  xc_conv2d_float_kw5xh2_stride_w3_opt(sd->out, sd->in, sd->kernels, sd->biases,
                                       sd->out_w, sd->out_h, sd->out_d,
                                       sd->in_w, sd->in_h, sd->in_d, *s, *e);
#endif
}
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<Beta_ConvF32OpData>(context);
  op_data->name = "XC_beta_convf32";

  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<Beta_ConvF32OpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  int out_d = output->dims->data[3];
  op_data->tc = xc_config->model_thread_count;
  calculateThreadSplit(op_data->tc, out_d, op_data->s, op_data->e);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<Beta_ConvF32OpData *>(node->user_data);
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *kernels =
      tflite::micro::GetEvalInput(context, node, 1);
  const TfLiteEvalTensor *bias = tflite::micro::GetEvalInput(context, node, 2);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  int out_w = output->dims->data[1];
  int out_h = output->dims->data[2];
  int out_d = output->dims->data[3];

  int in_w = input->dims->data[1];
  int in_h = input->dims->data[2];
  int in_d = input->dims->data[3];

  // Pointers to data in In/Out Tensors
  float *out_data = tflite::micro::GetTensorData<float>(output);
  float *in_data =
      const_cast<float *>(tflite::micro::GetTensorData<float>(input));
  float *kernel_data =
      const_cast<float *>(tflite::micro::GetTensorData<float>(kernels));
  float *bias_data =
      const_cast<float *>(tflite::micro::GetTensorData<float>(bias));

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());

  // todo - this second for-loop is unpleasant
  for (int t = 0; t < op_data->tc - 1; ++t) {
    thread_variable_setup(&op_data->s[t], &op_data->e[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }

  Beta_ConvF32Shared shared_data;
  shared_data.out = out_data;
  shared_data.in = in_data;
  shared_data.kernels = kernel_data;
  shared_data.biases = bias_data;
  shared_data.out_w = out_w;
  shared_data.out_h = out_h;
  shared_data.out_d = out_d;
  shared_data.in_w = in_w;
  shared_data.in_h = in_h;
  shared_data.in_d = in_d;

  // Now set up shared data, shared function pointer, and data for final thread.
  thread_call((void *)&shared_data, &op_data->s[op_data->tc - 1], &op_data->e[op_data->tc - 1],
              (thread_function_pointer_t)beta_convf32_thread_worker,
              &xc_config->thread_info);

  return kTfLiteOk;
}

} // namespace beta_convf32

TFLMRegistration *Register_XC_beta_convf32() {
  static TFLMRegistration r = {beta_convf32::Init, nullptr,
                                    beta_convf32::Prepare, beta_convf32::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
