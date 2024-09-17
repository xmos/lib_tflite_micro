// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "xcore_common.h"
#include "xcore_config.h"
#include "xcore_utils.h"
extern "C" {
#include "lib_nn/api/nn_operator.h"
#include "lib_nn/api/quadratic_interpolation.h"
}

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace lookup {

struct LookupShared {
  uint8_t *X;
  uint8_t *Y;
  uint8_t *table;
};

extern "C" {
void lookup8_thread_worker(void *shared, void *start, void *end) {
  int *s = static_cast<int *>(start);
  int *e = static_cast<int *>(end);
  auto sd = static_cast<LookupShared *>(shared);
  // lookup takes start and count instead of start and end
  lookup8(sd->Y, sd->X, sd->table, *s, *e - *s);
}

void lookup16_thread_worker(void *shared, void *start, void *end) {
  int *s = static_cast<int *>(start);
  int *e = static_cast<int *>(end);
  auto sd = static_cast<LookupShared *>(shared);
  // output and input pointers are adjusted with thread start
  quadratic_interpolation_128((int16_t *)sd->Y + *s, (int16_t *)sd->X + *s,
                              sd->table, *e - *s);
}
}
// This is the struct that contains the data required by the operator
struct LookupOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int tc;
  int s[XCORE_MAX_NUM_THREADS];
  int e[XCORE_MAX_NUM_THREADS];
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<LookupOpData>(context);
  op_data->name = "XC_lookup";
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<LookupOpData *>(node->user_data);
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

  auto *op_data = static_cast<LookupOpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite_micro::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *table = tflite_micro::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  const uint8_t *table_vals = tflite_micro::micro::GetTensorData<uint8_t>(table);
  uint8_t *out_data = tflite_micro::micro::GetTensorData<uint8_t>(output);
  const uint8_t *in_data = tflite_micro::micro::GetTensorData<uint8_t>(input);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const int tc = op_data->tc;
  LookupShared shared_data;
  shared_data.Y = out_data;
  shared_data.X = const_cast<uint8_t *>(in_data);
  shared_data.table = const_cast<uint8_t *>(table_vals);
  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&op_data->s[t], (void *)&op_data->e[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }

  thread_function_pointer_t fn;
  switch (input->type) {
  case kTfLiteInt8: {
    fn = lookup8_thread_worker;
    break;
  }
  case kTfLiteInt16: {
    fn = lookup16_thread_worker;
    break;
  }
  default: {
    return kTfLiteError;
  }
  }

  thread_call((void *)&shared_data, &op_data->s[tc - 1], &op_data->e[tc - 1],
              (thread_function_pointer_t)fn, &xc_config->thread_info);
  return kTfLiteOk;
}

} // namespace lookup

TFLMRegistration *Register_XC_lookup() {
  static TFLMRegistration r = {lookup::Init, nullptr, lookup::Prepare,
                               lookup::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
