// Copyright (c) 2022, XMOS Ltd, All rights reserved

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_interpreter.h"
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

#ifdef __xcore__
  chanend_t c_flash = (chanend_t) static_cast<int>(
      reinterpret_cast<intptr_t>(op_data->flash_data));
  chan_out_word(c_flash, 0); // TODO: share with aiserver.
  transacting_chanend_t t = chan_init_transaction_slave(c_flash);
  t_chan_out_word(&t, op_data->addr);
  t_chan_out_word(&t, op_data->size);
  for (int i = 0; i < op_data->size; i++) {
    data_ptr[i] = t_chan_in_byte(&t);
  }
  chan_complete_transaction(t);

  // load_from_flash_ll(op_data->c_flash, data_ptr, op_data->address,
  // op_data->bytes);
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

TfLiteRegistration *Register_XC_ld_flash() {
  static TfLiteRegistration r = {flash::Init, nullptr, flash::Prepare,
                                 flash::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
