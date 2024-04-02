// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
#include "nn_op_utils.h"
#include "vpu_memmove_word_aligned.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace concat {

inline void memcpy_wrapper(void *dst, void *src, size_t size) {
  memcpy(dst, src, size);
}

inline void vpu_memcpy_wrapper(void *dst, void *src, size_t size) {
  vpu_memmove_word_aligned(dst, src, size);
}

struct ConcatShared {
  int offset;
  int size1;
  int size2;
};

struct ConcatThread {
  const void *in_data1;
  const void *in_data2;
  void *out_data;
  int num_copies;
};

extern "C" {
void concat_thread_worker(void *shared, void *thread, void *nothing) {
  ConcatShared *shared_data = static_cast<ConcatShared *>(shared);
  ConcatThread *thread_data = static_cast<ConcatThread *>(thread);
  slice_memcpy_1d((int8_t *)thread_data->out_data,
                  (int8_t *)thread_data->in_data1, shared_data->size1,
                  shared_data->offset, thread_data->num_copies, memcpy_wrapper);
  slice_memcpy_1d((int8_t *)thread_data->out_data,
                  (int8_t *)thread_data->in_data2, shared_data->size2,
                  shared_data->offset, thread_data->num_copies, memcpy_wrapper);
}
}

struct ConcatOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int num_copies;
  int size1;
  int size2;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<ConcatOpData>(context);
  op_data->name = "XC_Concat";
  auto parser = CustomOptionParser(buffer, length);
  op_data->size1 = parser.parseNamedCustomOption("s1").AsInt32();
  op_data->size2 = parser.parseNamedCustomOption("s2").AsInt32();
  op_data->num_copies = parser.parseNamedCustomOption("n").AsInt32();
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<ConcatOpData *>(node->user_data);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  TFLITE_DCHECK(node->user_data != nullptr);

  auto *op_data = static_cast<ConcatOpData *>(node->user_data);
  // Get Input/Output Tensors
  const TfLiteEvalTensor *input1 =
      tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *input2 =
      tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  // Pointers to data in In/Out Tensors
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const void *in_data1 = tflite::micro::GetTensorData<void>(input1);
  const void *in_data2 = tflite::micro::GetTensorData<void>(input2);
  void *out_data = tflite::micro::GetTensorData<void>(output);
  const int32_t offset = op_data->size1 + op_data->size2;
  int total_copies = 0;
  ConcatThread thread_datas[XCORE_MAX_NUM_THREADS];
  for (int t = 0; t < xc_config->model_thread_count; t++) {
    const int num_copies =
        op_data->num_copies / xc_config->model_thread_count +
        (t < op_data->num_copies % xc_config->model_thread_count);
    thread_datas[t] = {(int8_t *)in_data1 + total_copies * op_data->size1,
                       (int8_t *)in_data2 + total_copies * op_data->size2,
                       (int8_t *)out_data + total_copies * offset, num_copies};
    total_copies += num_copies;
    if (t < xc_config->model_thread_count - 1)
      thread_variable_setup((void *)&thread_datas[t], NULL,
                            xc_config->thread_info.thread_ids.id[t]);
  }
  ConcatShared shared_data = {offset, op_data->size1, op_data->size2};
  thread_call((void *)&shared_data,
              &thread_datas[xc_config->model_thread_count - 1], NULL,
              (thread_function_pointer_t)concat_thread_worker,
              &xc_config->thread_info);
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
