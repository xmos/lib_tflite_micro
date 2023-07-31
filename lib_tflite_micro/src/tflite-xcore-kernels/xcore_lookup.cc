// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"
extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace lookup {

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //

struct LookupShared {
  uint8_t *X;
  uint8_t *Y;
  uint8_t *table;
};

int s[] = {0, 392, 784, 1176};
int c[] = {392, 392, 392, 392};

extern "C" {
// TODO
//#pragma stackfunction 1000
void lookup_thread_worker(void *shared, void *start, void *count) {
  int *s = static_cast<int *>(start);
  int *c = static_cast<int *>(count);
  auto sd = static_cast<LookupShared *>(shared);
  lookup8(sd->Y, sd->X, sd->table, *s, *c);
}
}

// This is the struct that contains the data required by the operator
struct LookupOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<LookupOpData>(context);
  op_data->name = "XC_lookup";

  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<LookupOpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  int input_size = 1;
  for (int i = 0; i < input->dims->size; i++)
    input_size *= input->dims->data[i];
  const TfLiteEvalTensor *table = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  uint8_t *table_vals =
      const_cast<uint8_t *>(tflite::micro::GetTensorData<uint8_t>(table));
  uint8_t *out_data = tflite::micro::GetTensorData<uint8_t>(output);
  uint8_t *in_data = const_cast<uint8_t *>(tflite::micro::GetTensorData<uint8_t>(input));

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());

  // for (int i = 0; i < input_size; i++) {
  //   ((int8_t *)out_data)[i] = ((int8_t *)table_vals)[((uint8_t *)in_val)[i]];
  // }

  //lookup8((uint8_t *)out_data, (uint8_t *)in_data, (uint8_t *)table_vals, 0, input_size);

  // todo - this second for-loop is unpleasant
  for (int t = 0; t < 4 - 1; ++t) {
    thread_variable_setup(&s[t], &c[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }

  LookupShared shared_data;
  shared_data.Y = out_data;
  shared_data.X = in_data;
  shared_data.table = table_vals;

  // Now set up shared data, shared function pointer, and data for final thread.
  thread_call((void *)&shared_data, &s[3], &c[3],
              (thread_function_pointer_t)lookup_thread_worker,
              &xc_config->thread_info);

  return kTfLiteOk;
}

} // namespace lookup

TfLiteRegistration_V1 *Register_XC_lookup() {
  static TfLiteRegistration_V1 r = {lookup::Init, nullptr, lookup::Prepare,
                                 lookup::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
