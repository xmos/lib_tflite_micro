// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace beta_activationf32 {

enum {
    NLA_ELU = 0,
    NLA_LOGISTICS = 1,
    NLA_TANH = 2,
};

void non_linear_activation(float *out, float *in, int type, int start, int end) {
    if (type == NLA_ELU) {
        for(int i = start; i < end; i++) {
            if (in[i] >= 0) {
                out[i] = in[i];
            } else {
                out[i] = expm1f(in[i]);
            }
        }
    } else if (type == NLA_TANH) {
        for(int i = start; i < end; i++) {
            out[i] = 2.0/(1.0 + expf(-in[i]*2))-1;
        }
    } else {
        assert(type == NLA_LOGISTICS);
        for(int i = start; i < end; i++) {
            out[i] = 1.0/(1.0 + expf(-in[i]));
        }
    }
}

// This is the struct that contains the data required by the operator
struct Beta_ActivationF32OpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int type;
};

struct Beta_ActivationF32Shared {
  float *out;
  float *in;
  int type;
};

extern "C" {
void beta_activationf32_thread_worker(void *shared, void *d_start, void *d_end) {
  int *s = static_cast<int *>(d_start);
  int *e = static_cast<int *>(d_end);
  auto sd = static_cast<Beta_ActivationF32Shared *>(shared);
  non_linear_activation(sd->out, sd->in, sd->type, *s, *e);
}
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<Beta_ActivationF32OpData>(context);
  op_data->name = "XC_beta_activationf32";

  auto parser = CustomOptionParser(buffer, length);
  op_data->type = parser.parseNamedCustomOption("type").AsInt32();

  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = static_cast<Beta_ActivationF32OpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  int input_size = tflite::micro::GetTensorShape(input).FlatSize();

  // Pointers to data in In/Out Tensors
  float *out_data = tflite::micro::GetTensorData<float>(output);
  float *in_data =
      const_cast<float *>(tflite::micro::GetTensorData<float>(input));

  int tc = 5;
  int split_start[5], split_end[5];
  int split_size = input_size;

  split_start[0] = 0;
  for (int i = 0; i < tc; i++) {
    auto split = (split_size + (tc - i) - 1) / (tc - i);
    split_size -= split;
    if (split > 0) {
      split_end[i] = split_start[i] + split;
      if (i != tc - 1)
        split_start[i + 1] = split_end[i];
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
    thread_variable_setup(&split_start[t], &split_end[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }

  Beta_ActivationF32Shared shared_data;
  shared_data.out = out_data;
  shared_data.in = in_data;
  shared_data.type = op_data->type;

  // Now set up shared data, shared function pointer, and data for final thread.
  thread_call((void *)&shared_data, &split_start[tc - 1], &split_end[tc - 1],
              (thread_function_pointer_t)beta_activationf32_thread_worker,
              &xc_config->thread_info);

  return kTfLiteOk;
}

} // namespace beta_activationf32

TfLiteRegistration_V1 *Register_XC_beta_activationf32() {
  static TfLiteRegistration_V1 r = {beta_activationf32::Init, nullptr,
                                    beta_activationf32::Prepare, beta_activationf32::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
