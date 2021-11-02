/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "Conv2d.hpp"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "xcore_custom_options.h"
#include "xcore_interpreter.h"
#include "xcore_dispatcher.h"
#include "xcore_utils.h"
extern "C" {
#include "nn_operator.h"
}
#ifdef __xcore__
#include <xcore/channel.h>
#include <xcore/channel_transaction.h>
#endif

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace flash {

// This is the struct that contains the data required to fully describe the work
// that the operator will perform. It needs to descibe the work for T threads.
// That means it must contain:
// - T sets of work, i.e. a list of jobs for each thread.
// - T scratch allocations, i.e. an amount of scratch memory for each thread.
struct FlashOpData : XCoreOpData {   // Inherits the operator name field from XCoreOpData
    uint32_t addr;
    uint32_t size;
    void  *flash_data;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(buffer != nullptr);
  
  auto op_data = construct_persistent_object<FlashOpData>(context);
  auto parser = CustomOptionParser(buffer, length);
  op_data->addr = parser.parseNamedCustomOption("addr").AsInt32();
  op_data->size   = parser.parseNamedCustomOption("size").AsInt32();
  tflite::micro::xcore::XCoreInterpreter* xint = reinterpret_cast<tflite::micro::xcore::XCoreInterpreter*>(context->impl_);
  op_data->flash_data = xint->flash_data;
  op_data->name    = "XC_Load_Flash";
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  int8_t *data_ptr = (int8_t*)tflite::micro::GetTensorData<int8_t>(output);
  auto* op_data = reinterpret_cast<FlashOpData*>(node->user_data);

#ifdef __xcore__
  chanend_t c_flash = (chanend_t) static_cast<int>(op_data->flash_data);
  chan_out_word(c_flash, 0);               // TODO: share with aiserver.
  transacting_chanend_t t = chan_init_transaction_slave(c_flash);
  chan_out_word(c_flash, op_data->addr);
  chan_out_word(c_flash, op_data->size);
  for(int i = 0; i < op_data->size; i++) {
      data_ptr[i] = t_chan_in_byte(&t);
  }
  chan_complete_transaction(t);

  // load_from_flash_ll(op_data->c_flash, data_ptr, op_data->address, op_data->bytes);
#else
  memcpy(data_ptr, ((int8_t *)op_data->flash_data) + op_data->addr, op_data->size);
#endif
  
  return kTfLiteOk;
}

}  // namespace flash

TfLiteRegistration* Register_LoadFromFlash() {
  static TfLiteRegistration r = {flash::Init, nullptr, flash::Prepare,
                                 flash::Eval};
  return &r;
}

}  // namespace xcore
}  // namespace micro
}  // namespace ops
}  // namespace tflite
