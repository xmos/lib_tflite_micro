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

struct LookupShared {
  uint8_t *X;
  uint8_t *Y;
  uint8_t *table;
};

extern "C" {
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
  int thread_count;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<LookupOpData>(context);
  op_data->name = "XC_lookup";
  auto parser = CustomOptionParser(buffer, length);
  op_data->thread_count = parser.parseNamedCustomOption("tc").AsInt32();

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
  const uint8_t *table_vals = tflite::micro::GetTensorData<uint8_t>(table);
  uint8_t *out_data = tflite::micro::GetTensorData<uint8_t>(output);
  const uint8_t *in_data = tflite::micro::GetTensorData<uint8_t>(input);
  // lookup8(out_data, in_data, table_vals, 0, input_size);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const int tc = op_data->thread_count;
  printf("threa_count: %d\n", op_data->thread_count);
  const int base_count = input_size / tc;
  const int extra = input_size % tc;
  int s[tc];
  int c[tc];
  LookupShared shared_data;
  shared_data.Y = out_data;
  shared_data.X = const_cast<uint8_t *>(in_data);
  shared_data.table = const_cast<uint8_t *>(table_vals);
  for (int t = 0; t < tc; t++) {
    s[t] = t * base_count + (t < extra ? t : extra);
    c[t] = base_count + (t < extra);
  }
  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&s[t], (void *)&c[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }
  thread_call((void *)&shared_data, &s[tc - 1], &c[tc - 1],
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
