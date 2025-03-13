// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"
extern "C" {
#include "nn_op_utils.h"
#include "lib_nn/api/nn_layers.h"
}

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace load_store_tensor {

// This is the struct that contains the data required by the operator
struct OpData {
  uint32_t addr;
  uint32_t size;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<OpData>(context);
  
  auto parser = CustomOptionParser(buffer, length);
  op_data->addr = parser.parseNamedCustomOption("a").AsInt32();
  op_data->size = parser.parseNamedCustomOption("s").AsInt32();

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  assert(true);
  return op_data;
}

TfLiteStatus Eval_Store(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = static_cast<OpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());

  const TfLiteEvalTensor *input = tflite_micro::micro::GetEvalInput(context, node, 0);
  const int8_t *data_ptr = tflite_micro::micro::GetTensorData<int8_t>(input);
  vpu_memcpy_ext(((int8_t *)xc_config->paging_ptr) + op_data->addr, data_ptr,
           op_data->size);
  return kTfLiteOk;
}

TfLiteStatus Eval_Load(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = static_cast<OpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());

  TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);
  int8_t *data_ptr = tflite_micro::micro::GetTensorData<int8_t>(output);
  
  int output_size = EvalTensorBytes(output);
  assert(output_size == op_data->size);
  
  vpu_memcpy_ext((void *)data_ptr,
           ((int8_t *)xc_config->paging_ptr) + op_data->addr,
           op_data->size);
  return kTfLiteOk;
}

} // namespace load_store_tensor

TFLMRegistration *Register_XC_store_tensor() {
  static TFLMRegistration r = {load_store_tensor::Init, nullptr, nullptr,
                                    load_store_tensor::Eval_Store};
  return &r;
}

TFLMRegistration *Register_XC_load_tensor() {
  static TFLMRegistration r = {load_store_tensor::Init, nullptr, nullptr,
                                    load_store_tensor::Eval_Load};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
