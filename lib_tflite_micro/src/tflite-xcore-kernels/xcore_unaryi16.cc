// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "xcore_common.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"
extern "C" {
#include "lib_nn/api/nn_operator.h"
#include "lib_nn/api/quantize_int16.h"
#include "lib_nn/api/multiply_int16.h"
#include "lib_nn/api/dequantize_int16.h"
}

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace unaryi16 {

const UnaryI16FnType fn_ptrs[] = {(UnaryI16FnType)&quantize_int16_tensor,
                                  (UnaryI16FnType)&requantize_int16_tensor,
                                  (UnaryI16FnType)&dequantize_int16_tensor};

struct UnaryI16Shared {
  int16_t *X;
  int16_t *Y;
  int16_t *blob;
  UnaryI16FnType fn;
  int inputTypeMultiplier;
  int outputTypeMultiplier;
};

extern "C" {
void unaryi16_thread_worker(void *shared, void *start, void *end) {
  int *s = static_cast<int *>(start);
  int *e = static_cast<int *>(end);
  auto sd = static_cast<UnaryI16Shared *>(shared);
  sd->fn(sd->Y + (*s * sd->outputTypeMultiplier),
         sd->X + (*s * sd->inputTypeMultiplier), *e - *s, sd->blob);
}
}

enum OpType {
  Quantize_t,
  Requantize_t,
  Dequantize_t,
};

// This is the struct that contains the data required by the operator
struct UnaryI16OpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int tc;
  int s[XCORE_MAX_NUM_THREADS];
  int e[XCORE_MAX_NUM_THREADS];
  int opType;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<UnaryI16OpData>(context);
  op_data->name = "XC_unaryi16";

  auto parser = CustomOptionParser(buffer, length);
  op_data->opType = parser.parseNamedCustomOption("type").AsInt32();
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<UnaryI16OpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const TfLiteEvalTensor *input = tflite_micro::micro::GetEvalInput(context, node, 0);
  int input_size = tflite_micro::micro::GetTensorShape(input).FlatSize();
  op_data->tc = xc_config->model_thread_count;
  calculateThreadSplit(op_data->tc, input_size, op_data->s, op_data->e);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = static_cast<UnaryI16OpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite_micro::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *blob = tflite_micro::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  const int16_t *in_data = tflite_micro::micro::GetTensorData<int16_t>(input);
  const int16_t *blob_data = tflite_micro::micro::GetTensorData<int16_t>(blob);
  int16_t *out_data = tflite_micro::micro::GetTensorData<int16_t>(output);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const int tc = op_data->tc;
  UnaryI16Shared shared_data;
  shared_data.Y = out_data;
  shared_data.X = const_cast<int16_t *>(in_data);
  shared_data.blob = const_cast<int16_t *>(blob_data);
  shared_data.fn = fn_ptrs[op_data->opType];
  shared_data.inputTypeMultiplier = op_data->opType == Quantize_t ? 2 : 1;
  shared_data.outputTypeMultiplier = op_data->opType == Dequantize_t ? 2 : 1;
  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&op_data->s[t], (void *)&op_data->e[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }
  thread_call((void *)&shared_data, &op_data->s[tc - 1], &op_data->e[tc - 1],
              (thread_function_pointer_t)unaryi16_thread_worker,
              &xc_config->thread_info);
  
  return kTfLiteOk;
}

} // namespace unaryi16

TFLMRegistration *Register_XC_unaryi16() {
  static TFLMRegistration r = {unaryi16::Init, nullptr, unaryi16::Prepare,
                               unaryi16::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
