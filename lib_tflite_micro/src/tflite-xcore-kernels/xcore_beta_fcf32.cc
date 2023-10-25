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
namespace beta_fcf32 {

// This is the struct that contains the data required by the operator
struct Beta_FcF32OpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int tc;
  int s[XCORE_MAX_NUM_THREADS];
  int e[XCORE_MAX_NUM_THREADS];
};

struct Beta_FcF32Shared {
  float *out;
  float *in;
  float *kernels;
  int out_f;
  int in_f;
};

extern "C" {
void beta_fcf32_thread_worker(void *shared, void *start, void *end) {
  int *s = static_cast<int *>(start);
  int *e = static_cast<int *>(end);
  auto sd = static_cast<Beta_FcF32Shared *>(shared);
  #ifdef NN_USE_REF
  xc_fc_float_ref(sd->out, sd->in, sd->kernels, sd->out_f,
                  sd->in_f);
  #else
  xc_fc_float_opt(sd->out, sd->in, sd->kernels, sd->out_f,
                  sd->in_f, *s, *e);
  #endif
}
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<Beta_FcF32OpData>(context);
  op_data->name = "XC_beta_fcf32";

  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<Beta_FcF32OpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  int out_f = output->dims->data[1];
  op_data->tc = xc_config->model_thread_count;
  calculateThreadSplit(op_data->tc, out_f, op_data->s, op_data->e);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<Beta_FcF32OpData *>(node->user_data);
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *kernels =
      tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  int out_f = output->dims->data[1];

  int in_f = input->dims->data[1];

  // Pointers to data in In/Out Tensors
  float *out_data = tflite::micro::GetTensorData<float>(output);
  float *in_data =
      const_cast<float *>(tflite::micro::GetTensorData<float>(input));
  float *kernel_data =
      const_cast<float *>(tflite::micro::GetTensorData<float>(kernels));

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());

  // todo - this second for-loop is unpleasant
  for (int t = 0; t < op_data->tc - 1; ++t) {
    thread_variable_setup(&op_data->s[t], &op_data->e[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }

  Beta_FcF32Shared shared_data;
  shared_data.out = out_data;
  shared_data.in = in_data;
  shared_data.kernels = kernel_data;
  shared_data.out_f = out_f;
  shared_data.in_f = in_f;

  // Now set up shared data, shared function pointer, and data for final thread.
  thread_call((void *)&shared_data, &op_data->s[op_data->tc - 1], &op_data->e[op_data->tc - 1],
              (thread_function_pointer_t)beta_fcf32_thread_worker,
              &xc_config->thread_info);

  return kTfLiteOk;
}

} // namespace beta_fcf32

TfLiteRegistration_V1 *Register_XC_beta_fcf32() {
  static TfLiteRegistration_V1 r = {beta_fcf32::Init, nullptr,
                                    beta_fcf32::Prepare, beta_fcf32::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
