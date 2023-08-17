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
namespace beta_transposeconvf32 {

// This is the struct that contains the data required by the operator
struct Beta_TransposeConvF32OpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
};

struct Beta_TransposeConvF32Shared {
  float *out;
  float *in;
  float *kernels;
  float *biases;
  int out_w, out_h, out_d;
  int in_w, in_h, in_d;
};

extern "C" {
void beta_transposeconvf32_thread_worker(void *shared, void *d_start,
                                         void *d_end) {
  int *s = static_cast<int *>(d_start);
  int *e = static_cast<int *>(d_end);
  auto sd = static_cast<Beta_TransposeConvF32Shared *>(shared);
#ifdef NN_USE_REF
  xc_transpose_conv2d_float_kw5xh2_stride_h3_ref(
      sd->out, sd->in, sd->kernels, sd->biases, sd->out_w, sd->out_h, sd->out_d,
      sd->in_w, sd->in_h, sd->in_d);
#else
  xc_transpose_conv2d_float_kw5xh2_stride_h3_opt(
      sd->out, sd->in, sd->kernels, sd->biases, sd->out_w, sd->out_h, sd->out_d,
      sd->in_w, sd->in_h, sd->in_d, *s, *e);
#endif
}
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data =
      construct_persistent_object<Beta_TransposeConvF32OpData>(context);
  op_data->name = "XC_beta_transposeconvf32";

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

  int tc = 5;
  int out_d_start[5], out_d_end[5];
  int out_depth_size = out_d;

  out_d_start[0] = 0;
  for (int i = 0; i < tc; i++) {
    auto split = (out_depth_size + (tc - i) - 1) / (tc - i);
    out_depth_size -= split;
    if (split > 0) {
      out_d_end[i] = out_d_start[i] + split;
      if (i != tc - 1)
        out_d_start[i + 1] = out_d_end[i];
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
    thread_variable_setup(&out_d_start[t], &out_d_end[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }

  Beta_TransposeConvF32Shared shared_data;
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
  thread_call((void *)&shared_data, &out_d_start[tc - 1], &out_d_end[tc - 1],
              (thread_function_pointer_t)beta_transposeconvf32_thread_worker,
              &xc_config->thread_info);

  return kTfLiteOk;
}

} // namespace beta_transposeconvf32

TfLiteRegistration_V1 *Register_XC_beta_transposeconvf32() {
  static TfLiteRegistration_V1 r = {beta_transposeconvf32::Init, nullptr,
                                    beta_transposeconvf32::Prepare,
                                    beta_transposeconvf32::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
