// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"

#ifdef __xcore__
#include <xcore/channel.h>
#include <xcore/channel_transaction.h>
extern "C" {
#include "memory_parallel_transport.h"
#include "nn_op_utils.h"
#include "load_weights.h"
}
#endif

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace load_weights {

enum OpType {
  Sync,
  DDR,
  Async,
};

constexpr int kMaxOutputNum = 10; // Maximum number of output tensors

// This is the struct that contains the data required to fully describe the work
// that the operator will perform.
struct FlashOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  uint32_t addr;
  uint32_t sizes[kMaxOutputNum];
  uint32_t op_type;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  TFLITE_DCHECK(buffer != nullptr);

  auto op_data = construct_persistent_object<FlashOpData>(context);
  auto parser = CustomOptionParser(buffer, length);
  op_data->addr = parser.parseNamedCustomOption("a").AsInt32();
  auto sizes_vec = parser.parseNamedCustomOption("s").AsVector();
  TFLITE_DCHECK(sizes_vec.size() <= kMaxOutputNum);

  for (int i = 0; i < sizes_vec.size(); i++) {
    op_data->sizes[i] = sizes_vec[i].AsInt32();
  }

  op_data->op_type = parser.parseNamedCustomOption("t").AsInt32();

  if (op_data->op_type == OpType::Async) {
    op_data->name = "XC_Load_Weights_Async";
  } else if (op_data->op_type == OpType::DDR) {
    op_data->name = "XC_Load_Weights_DDR";
  } else {
    op_data->name = "XC_Load_Weights_Sync";
  }
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = reinterpret_cast<FlashOpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
#ifdef __xcore__
  // If DDR, we can do a direct copy with the VPU
  // If not DDR, the weights will be in flash or on another tile
  if (op_data->op_type == OpType::DDR) {
    assert(node->outputs->size == 1 && "DDR loads have only one output!");
    TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);
    int8_t *data_ptr = tflite_micro::micro::GetTensorData<int8_t>(output);
    vpu_memcpy_ext((void *)data_ptr,
                   ((int8_t *)xc_config->weights_data_ptr) + op_data->addr,
                   op_data->sizes[0]);
  } else {

    chanend_t c_flash_or_tile = (chanend_t) static_cast<int>(
        reinterpret_cast<intptr_t>(xc_config->weights_data_ptr));

#define MAX_OUTPUTS 4
    int *data_ptrs[MAX_OUTPUTS];
    int data_sizes_in_words[MAX_OUTPUTS];

    assert(node->outputs->size < MAX_OUTPUTS);
    for (int i = 0; i < node->outputs->size; ++i) {
      TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, i);
      data_ptrs[i] = (int*)tflite_micro::micro::GetTensorData<int8_t>(output);
      data_sizes_in_words[i] = op_data->sizes[i]/4;
    }

    if (op_data->op_type == OpType::Async) {
      load_weights_asynchronous(c_flash_or_tile, data_ptrs, data_sizes_in_words,
                              node->outputs->size, op_data->addr);
    } else {
      thread_info_t *tif = &xc_config->thread_info;
      load_weights_synchronous(c_flash_or_tile, data_ptrs, data_sizes_in_words,
                              node->outputs->size, op_data->addr, xc_config->model_thread_count, tif);
    }    
  }

#else
  int addr_offset = 0;

  for (int i = 0; i < node->outputs->size; ++i) {
    TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, i);
    int8_t *data_ptr = tflite_micro::micro::GetTensorData<int8_t>(output);
    memcpy((void *)data_ptr,
           ((int8_t *)xc_config->weights_data_ptr) + op_data->addr +
               addr_offset,
           op_data->sizes[i]);
    addr_offset += op_data->sizes[i];
  }
#endif

  return kTfLiteOk;
}

TfLiteStatus Eval_Wait(TfLiteContext *context, TfLiteNode *node) {
#ifdef __xcore__
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
 chanend_t c_flash_or_tile = (chanend_t) static_cast<int>(
        reinterpret_cast<intptr_t>(xc_config->weights_data_ptr));
  load_weights_asynchronous_wait(c_flash_or_tile);
#endif

  return kTfLiteOk;
}

} // namespace load_weights

TFLMRegistration *Register_XC_ld_weights() {
  static TFLMRegistration r = {load_weights::Init, nullptr,
                               load_weights::Prepare, load_weights::Eval};
  return &r;
}

TFLMRegistration *Register_XC_ld_weights_wait() {
  static TFLMRegistration r = {nullptr, nullptr,
                               nullptr, load_weights::Eval_Wait};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
