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
#endif

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace flash {

constexpr int kMaxOutputNum = 10; // Maximum number of output tensors

// This is the struct that contains the data required to fully describe the work
// that the operator will perform.
struct FlashOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  uint32_t addr;
  uint32_t sizes[kMaxOutputNum];
  void *flash_data;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  TFLITE_DCHECK(buffer != nullptr);

  auto op_data = construct_persistent_object<FlashOpData>(context);
  auto parser = CustomOptionParser(buffer, length);
  op_data->addr = parser.parseNamedCustomOption("addr").AsInt32();
  auto sizes_vec = parser.parseNamedCustomOption("sizes").AsVector();
  TFLITE_DCHECK(sizes_vec.size() <= kMaxOutputNum);

  for (int i = 0; i < sizes_vec.size(); i++) {
    op_data->sizes[i] = sizes_vec[i].AsInt32();
  }

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  op_data->flash_data = xc_config->flash_data;
  op_data->name = "XC_Load_Flash";
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
  thread_info_t *tif = &xc_config->thread_info;
#ifdef __xcore__
  chanend_t c_flash = (chanend_t) static_cast<int>(
      reinterpret_cast<intptr_t>(op_data->flash_data));
  chan_out_word(c_flash, 0); // TODO: share with aiserver.
  chan_out_word(c_flash, op_data->addr);

  // Any latency with flash might cause dropping of words.
  // We initialize the data_ptrs here so that they are ready
  // before we enter the flash data read loop.
#define MAX_OUTPUTS 4
  int8_t *data_ptrs[MAX_OUTPUTS];
  int8_t *data_ptr;
  assert(node->outputs->size < MAX_OUTPUTS);
  for (int i = 0; i < node->outputs->size; ++i) {
    TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, i);
    data_ptrs[i] = tflite::micro::GetTensorData<int8_t>(output);
  }
    
  int32_t total_size = 0;
  for (int i = 0; i < node->outputs->size; ++i) {
    total_size += op_data->sizes[i];
  }
  chan_out_word(c_flash, total_size);

  for (int i = 0; i < node->outputs->size; ++i) {
    data_ptr = data_ptrs[i];

    int use_parallel_mode = chan_in_word(c_flash);
    if (use_parallel_mode) {
      memory_parallel_receive_thread_call(c_flash, data_ptr, op_data->sizes[i], tif);
    } else {
      // The sizes are in bytes and we read from flash in words
      int op_data_size_in_words = op_data->sizes[i]/4;
      //#pragma clang loop unroll_count(4)
      for (int j = 0; j < op_data_size_in_words; j++) {
        // We are reading directly from flash chanend here.
        // We use chanend_in_word() instead of chan_in_word() to
        // avoid handshake.
        // Adding something like a printf() within this loop
        // might slow it down enough to corrupt the received data.
        ((uint32_t *)data_ptr)[j] = chanend_in_word(c_flash);
      }
    }
  }
  // As there is no handshake, we have to accept the end token
  // to close the chanend
  chanend_check_end_token(c_flash);

#else
  int addr_offset = 0;
  for (int i = 0; i < node->outputs->size; ++i) {
    TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, i);
    int8_t *data_ptr = tflite::micro::GetTensorData<int8_t>(output);
    memcpy((void *)data_ptr,
           ((int8_t *)op_data->flash_data) + op_data->addr + addr_offset,
           op_data->sizes[i]);
    addr_offset += op_data->sizes[i];
  }
#endif

  return kTfLiteOk;
}

} // namespace flash

TfLiteRegistration_V1 *Register_XC_ld_flash() {
  static TfLiteRegistration_V1 r = {flash::Init, nullptr, flash::Prepare,
                                 flash::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
