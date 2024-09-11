// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "thread_call.h"
#include "xcore_common.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"
#include <string.h>

extern "C" {
#include "vpu_memmove_word_aligned.h"
}

constexpr int kMaxNumInputs = 13;

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace concat {

using tflite::micro::GetEvalInput;
using tflite::micro::GetEvalOutput;
using tflite::micro::GetTensorData;

struct ConcatOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int32_t num_copies;
  int32_t sizes[kMaxNumInputs];
  int32_t num_inputs;
  int tc;
  int s[XCORE_MAX_NUM_THREADS];
  int e[XCORE_MAX_NUM_THREADS];
  int32_t total_size;
  bool all_ones;
  void (*func_ptr)(void *, const void *, unsigned);
};

struct ConcatShared {
  const int8_t *inputs[kMaxNumInputs];
  int8_t *output;
  ConcatOpData *op_data;
};

extern "C" {
void concat_thread_worker(void *shared, void *start, void *end) {
  int *s = static_cast<int *>(start);
  int *e = static_cast<int *>(end);
  auto sd = static_cast<ConcatShared *>(shared);
  auto out_data = sd->output;
  auto op_data = sd->op_data;
  auto inputs = sd->inputs;
  auto out_start = out_data + (*s) * op_data->total_size;
  const int8_t *input_starts[kMaxNumInputs];
  for (int i = 0; i < op_data->num_inputs; i++)
    input_starts[i] = inputs[i] + (*s) * op_data->sizes[i];
  if (op_data->all_ones) {
    for (int i = 0; i < (*e) - (*s); i++) {
      for (int j = 0; j < op_data->num_inputs; j++) {
        *out_start++ = *input_starts[j]++;
      }
    }
    return;
  }
  void (*func_ptr)(void *, const void *, unsigned) = op_data->func_ptr;
  for (int i = 0; i < (*e - *s); i++) {
    for (int j = 0; j < op_data->num_inputs; j++) {
      const int size = op_data->sizes[j];
      func_ptr(out_start, input_starts[j], size);
      out_start += size;
      input_starts[j] += size;
    }
  }
}
}

void memmove_wrapper(void *dst, const void *src, unsigned size) {
  memmove(dst, src, size);
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<ConcatOpData>(context);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  op_data->tc = xc_config->model_thread_count;
  op_data->name = "XC_Concat";
  auto parser = CustomOptionParser(buffer, length);
  op_data->num_copies = parser.parseNamedCustomOption("n").AsInt32();
  calculateThreadSplit(op_data->tc, op_data->num_copies, op_data->s,
                       op_data->e);
  op_data->num_inputs = parser.parseNamedCustomOption("i").AsInt32();
  auto sizes = parser.parseNamedCustomOption("s").AsVector();
  TFLITE_DCHECK(op_data->num_inputs <= kMaxNumInputs);
  bool all_ones = true;
  op_data->total_size = 0;
  for (int i = 0; i < op_data->num_inputs; i++) {
    int32_t s = sizes[i].AsInt32();
    op_data->sizes[i] = s;
    op_data->total_size += s;
    if (s != 1)
      all_ones = false;
  }
  op_data->all_ones = all_ones;
  bool use_vpu = parser.parseNamedCustomOption("v").AsBool();
  op_data->func_ptr = use_vpu ? vpu_memmove_word_aligned : memmove_wrapper;
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  TFLITE_DCHECK(node->user_data != nullptr);

  auto *op_data = static_cast<ConcatOpData *>(node->user_data);
  const int8_t *inputs[kMaxNumInputs];
  for (int i = 0; i < op_data->num_inputs; i++)
    inputs[i] = GetTensorData<int8_t>(GetEvalInput(context, node, i));

  TfLiteEvalTensor *output = GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  int8_t *out_data = GetTensorData<int8_t>(output);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const int tc = op_data->tc;
  ConcatShared shared_data;
  shared_data.op_data = op_data;
  shared_data.output = out_data;
  for (int i = 0; i < op_data->num_inputs; i++)
    shared_data.inputs[i] = inputs[i];

  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&op_data->s[t], (void *)&op_data->e[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }
  thread_call((void *)&shared_data, &op_data->s[tc - 1], &op_data->e[tc - 1],
              concat_thread_worker, &xc_config->thread_info);
  return kTfLiteOk;
}

} // namespace concat

TFLMRegistration *Register_XC_concat() {
  static TFLMRegistration r = {concat::Init, nullptr, concat::Prepare,
                               concat::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
