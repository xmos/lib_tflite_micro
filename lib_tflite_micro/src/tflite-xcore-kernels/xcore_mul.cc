// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"
extern "C" {
#include "lib_nn/api/nn_layers.h"
#include "lib_nn/api/nn_operator.h"
}

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace mul {

struct MulShared {
  int8_t *Y;
  int8_t *X1;
  int8_t *X2;
  nn_mul_params_t *blob;
};

extern "C" {
void mul_thread_worker(void *shared, void *start, void *end) {
  int *s = static_cast<int *>(start);
  int *e = static_cast<int *>(end);
  auto sd = static_cast<MulShared *>(shared);
  mul_elementwise(sd->X1 + *s, sd->X2 + *s, *e - *s, sd->blob, sd->Y + *s);
}
}

// This is the struct that contains the data required by the operator
struct MulOpData {
  nn_mul_params_t *mp_params;
  int tc;
  int s[XCORE_MAX_NUM_THREADS];
  int e[XCORE_MAX_NUM_THREADS];
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<MulOpData>(context);

  auto parser = CustomOptionParser(buffer, length);
  op_data->mp_params = (nn_mul_params_t *)parser.parseNamedCustomOption("mp").AsBlob().data();

  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = static_cast<MulOpData *>(node->user_data);
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

  auto *op_data = static_cast<MulOpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input1 =
      tflite_micro::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *input2 =
      tflite_micro::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  int8_t *out_data = tflite_micro::micro::GetTensorData<int8_t>(output);
  const int8_t *in1_data = tflite_micro::micro::GetTensorData<int8_t>(input1);
  const int8_t *in2_data = tflite_micro::micro::GetTensorData<int8_t>(input2);

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const int tc = op_data->tc;
  MulShared shared_data;
  shared_data.Y = out_data;
  shared_data.X1 = const_cast<int8_t *>(in1_data);
  shared_data.X2 = const_cast<int8_t *>(in2_data);
  shared_data.blob = op_data->mp_params;
  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&op_data->s[t], (void *)&op_data->e[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }
  thread_call((void *)&shared_data, &op_data->s[tc - 1], &op_data->e[tc - 1],
              (thread_function_pointer_t)mul_thread_worker,
              &xc_config->thread_info);

  return kTfLiteOk;
}

} // namespace mul

TFLMRegistration *Register_XC_mul() {
  static TFLMRegistration r = {mul::Init, nullptr, mul::Prepare,
                                    mul::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
