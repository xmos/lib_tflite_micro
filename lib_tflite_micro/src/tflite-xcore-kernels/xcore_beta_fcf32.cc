// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "conv2d_float.h"
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
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
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

  int tc = 5;
  int out_f_start[5], out_f_end[5];
  int out_f_size = out_f;

  out_f_start[0] = 0;
  for (int i = 0; i < tc; i++) {
    auto split = (out_f_size + (tc - i) - 1) / (tc - i);
    out_f_size -= split;
    if (split > 0) {
      out_f_end[i] = out_f_start[i] + split;
      if (i != tc - 1)
        out_f_start[i + 1] = out_f_end[i];
    } else {
      tc = i;
      break;
    }
  }

  // for(int i = 0; i < tc; i++) {
  //   printf("\ns = %d, e = %d", out_d_start[i], out_d_end[i]);
  // }
  // printf("\n\n");

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());

  // todo - this second for-loop is unpleasant
  for (int t = 0; t < tc - 1; ++t) {
    thread_variable_setup(&out_f_start[t], &out_f_end[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }

  Beta_FcF32Shared shared_data;
  shared_data.out = out_data;
  shared_data.in = in_data;
  shared_data.kernels = kernel_data;
  shared_data.out_f = out_f;
  shared_data.in_f = in_f;

  // Now set up shared data, shared function pointer, and data for final thread.
  thread_call((void *)&shared_data, &out_f_start[tc - 1], &out_f_end[tc - 1],
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
