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
namespace add {

struct AddShared {
  int8_t *Y;
  int8_t *X1;
  int8_t *X2;
  nn_add_params_t *blob;
};

extern "C" {
void add_thread_worker(void *shared, void *start, void *end) {
  int *s = static_cast<int *>(start);
  int *e = static_cast<int *>(end);
  auto sd = static_cast<AddShared *>(shared);
  add_elementwise(sd->Y, sd->X1, sd->X2, sd->blob, *s, *e - *s);
}
}

// This is the struct that contains the data required by the operator
struct AddOpData {
  nn_add_params_t params;
  int tc;
  int s[XCORE_MAX_NUM_THREADS];
  int e[XCORE_MAX_NUM_THREADS];
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<AddOpData>(context);

  auto parser = CustomOptionParser(buffer, length);
  int m1 = parser.parseNamedCustomOption("m1").AsInt32();
  int m2 = parser.parseNamedCustomOption("m2").AsInt32();
  int bias = parser.parseNamedCustomOption("bias").AsInt32();
  int shift = parser.parseNamedCustomOption("shift").AsInt32();

  // Broadcast values into vectors
  // We are VLMACC-ing in 16-bit mode
  for (int i = 0; i < VPU_INT16_VLMACC_ELMS; i++) {
    op_data->params.m1[i] = (int16_t)m1;
    op_data->params.m2[i] = (int16_t)m2;
    op_data->params.shift[i] = (int16_t)shift;
    // Split 32-bit bias into two 16-bit values
    op_data->params.bias_hi[i] = bias >> 16;
    op_data->params.bias_lo[i] = (int16_t) (bias & 0XFFFF);
  }

  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = static_cast<AddOpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const TfLiteEvalTensor *output =
      tflite_micro::micro::GetEvalOutput(context, node, 0);
  int output_size = tflite_micro::micro::GetTensorShape(output).FlatSize();
  op_data->tc = calculateAlignedThreadSplit(xc_config->model_thread_count, output_size, op_data->s, op_data->e);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<AddOpData *>(node->user_data);
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input1 =
      tflite_micro::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *input2 =
      tflite_micro::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  int8_t *in1_data =
      const_cast<int8_t *>(tflite_micro::micro::GetTensorData<int8_t>(input1));
  int8_t *in2_data =
      const_cast<int8_t *>(tflite_micro::micro::GetTensorData<int8_t>(input2));
  int8_t *out_data = tflite_micro::micro::GetTensorData<int8_t>(output);

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const int tc = op_data->tc;
  AddShared shared_data;
  shared_data.Y = out_data;
  shared_data.X1 = in1_data;
  shared_data.X2 = in2_data;
  shared_data.blob = &op_data->params;
  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&op_data->s[t], (void *)&op_data->e[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }
  thread_call((void *)&shared_data, &op_data->s[tc - 1], &op_data->e[tc - 1],
              (thread_function_pointer_t)add_thread_worker,
              &xc_config->thread_info);

  return kTfLiteOk;
}

} // namespace add

TFLMRegistration *Register_XC_add() {
  static TFLMRegistration r = {add::Init, nullptr, add::Prepare, add::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
