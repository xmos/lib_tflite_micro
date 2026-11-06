// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "xcore_custom_options.h"
#include "../thread_call.h"
#include "xcore_config.h"
#include "xcore_utils.h"
extern "C" {
#include "lib_nn/api/nn_operator.h"
#include "lib_nn/api/nn_layers.h"
}

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace mean {

typedef struct MeanWorkerArg0 {
  int input_offset;
  int output_offset;
  int start_dim_size;
};

// This is the struct that contains the data required by the operator
struct MeanOpData {
  int start_dim_size;
  int mean_dim_size;
  int end_dim_size;
  float in_zero_point;
  float out_zero_point;
  float scale_mul;
  int tc;
  MeanWorkerArg0 arg0[XCORE_MAX_NUM_THREADS];
};

struct MeanShared {
  int8_t *input;
  int8_t *output;
  MeanOpData *op_data;
};

extern "C" {
void mean_int8_thread_worker(void *shared, void *arg0, void *arg1) {
  MeanWorkerArg0 *arg = static_cast<MeanWorkerArg0 *>(arg0);
  (void) arg1;
  auto sd = static_cast<MeanShared *>(shared);
  int8_t *input = &sd->input[arg->input_offset];
  int8_t *output = &sd->output[arg->output_offset];
  mean_int8(input, output, arg->start_dim_size, sd->op_data->mean_dim_size,
            sd->op_data->end_dim_size, sd->op_data->in_zero_point,
            sd->op_data->out_zero_point, sd->op_data->scale_mul);
}
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<MeanOpData>(context);

  auto parser = CustomOptionParser(buffer, length);
  op_data->start_dim_size = parser.parseNamedCustomOption("s").AsInt32();
  op_data->mean_dim_size = parser.parseNamedCustomOption("m").AsInt32();
  op_data->end_dim_size = parser.parseNamedCustomOption("e").AsInt32();
  op_data->in_zero_point = parser.parseNamedCustomOption("i").AsFloat();
  op_data->out_zero_point = parser.parseNamedCustomOption("o").AsFloat();
  op_data->scale_mul = parser.parseNamedCustomOption("sm").AsFloat();
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<MeanOpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  int s[XCORE_MAX_NUM_THREADS];
  int e[XCORE_MAX_NUM_THREADS];
  op_data->tc = calculateAlignedThreadSplit(
    xc_config->model_thread_count, op_data->start_dim_size, s, e);
  // Turn start and end into input and output offset
  for (int t = 0; t < op_data->tc; ++t) {
    op_data->arg0[t].input_offset = s[t] * op_data->mean_dim_size * op_data->end_dim_size;
    op_data->arg0[t].output_offset = s[t] * op_data->end_dim_size;
    op_data->arg0[t].start_dim_size = e[t]-s[t];
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<MeanOpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite_micro::micro::GetEvalInput(context, node, 0);

  TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  int8_t *out_data = tflite_micro::micro::GetTensorData<int8_t>(output);
  const int8_t *in_data = tflite_micro::micro::GetTensorData<int8_t>(input);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const int tc = op_data->tc;
  if (tc == 1 && input->type == kTfLiteInt8) {
    mean_int8(in_data, out_data, op_data->start_dim_size, op_data->mean_dim_size,
              op_data->end_dim_size, op_data->in_zero_point,
              op_data->out_zero_point, op_data->scale_mul);
    return kTfLiteOk;
  }
  MeanShared shared_data;
  shared_data.input = const_cast<int8_t *>(in_data);
  shared_data.output = out_data;
  shared_data.op_data = op_data;
  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&op_data->arg0[t], nullptr,
                          xc_config->thread_info.thread_ids.id[t]);
  }

  thread_function_pointer_t fn;
  switch (input->type) {
  case kTfLiteInt8: {
    fn = mean_int8_thread_worker;
    break;
  }
  default: {
    return kTfLiteError;
  }
  }

  thread_call((void *)&shared_data, &op_data->arg0[tc - 1], nullptr,
              (thread_function_pointer_t)fn, &xc_config->thread_info);

  return kTfLiteOk;
}

} // namespace mean

TFLMRegistration *Register_XC_mean() {
  static TFLMRegistration r = {mean::Init, nullptr, mean::Prepare, mean::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
