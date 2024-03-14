// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "xcore_common.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"
extern "C" {
#include "lib_nn/api/nn_operator.h"
#include "lib_nn/api/quantize_int16.h"
#include "lib_nn/api/multiply_int16.h"
#include "lib_nn/api/dequantize_int16.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace blob_unaryi16 {

const UnaryI16FnType fn_ptrs[] = {(UnaryI16FnType)&quantize_int16_tensor,
                                  (UnaryI16FnType)&requantize_int16_tensor,
                                  (UnaryI16FnType)&dequantize_int16_tensor};

struct Blob_UnaryI16Shared {
  int16_t *X;
  int16_t *Y;
  uint8_t *op_blob;
  UnaryI16FnType fn;
  int inputTypeMultiplier;
  int outputTypeMultiplier;
};

//quantize_int16_wrapper()
//fn_ptr(tensor_descriptor, parallel_descriptor, operation_descriptor)

extern "C" {

void xc_unaryi16_invoke(int32_t **tensor_descriptor, uint8_t *op_descriptor, int32_t *th_descriptor,
                 int thread_num) {
    int tc = th_descriptor[0];
    if(thread_num >= tc)
      return;

    int16_t* in = (int16_t*)tensor_descriptor[0];
    float* out = (float*)tensor_descriptor[1];
    int *thread_data = (int*)(th_descriptor + 1);
    op_descriptor = op_descriptor + 4;

    int th_start = thread_data[thread_num];
    int th_count = thread_data[thread_num + tc];

    dequantize_int16_tensor(out + th_start, in + th_start, th_count, op_descriptor);
}

void blob_unaryi16_thread_worker(void *shared, void *start, void *count) {
  int *s = static_cast<int *>(start);
  int *c = static_cast<int *>(count);
  auto sd = static_cast<Blob_UnaryI16Shared *>(shared);
  sd->fn(sd->Y + (*s * sd->outputTypeMultiplier),
         sd->X + (*s * sd->inputTypeMultiplier), *c, sd->op_blob);
}
}

enum OpType {
  Quantize_t,
  Requantize_t,
  Dequantize_t,
};

// This is the struct that contains the data required by the operator
struct Blob_UnaryI16OpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int tc;
  int s[XCORE_MAX_NUM_THREADS];
  int e[XCORE_MAX_NUM_THREADS];
  int opType;
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  // Get Input/Output Tensors
  const TfLiteEvalTensor *op_blob = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *th_blob = tflite::micro::GetEvalInput(context, node, 1);
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 2);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  const uint8_t *op_blob_data = tflite::micro::GetTensorData<uint8_t>(op_blob);
  const int32_t *th_blob_data = tflite::micro::GetTensorData<int32_t>(th_blob);
  const int16_t *in_data = tflite::micro::GetTensorData<int16_t>(input);
  int16_t *out_data = tflite::micro::GetTensorData<int16_t>(output);
  
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  
  int op_type = op_blob_data[0]; 
  int tc = th_blob_data[0];
  int *thread_data = (int*)(th_blob_data + 1);
  
  Blob_UnaryI16Shared shared_data;
  shared_data.Y = out_data;
  shared_data.X = const_cast<int16_t *>(in_data);
  shared_data.op_blob = const_cast<uint8_t *>(op_blob_data + 4);
  shared_data.fn = fn_ptrs[op_type];
  shared_data.inputTypeMultiplier = op_type == Quantize_t ? 2 : 1;
  shared_data.outputTypeMultiplier = op_type == Dequantize_t ? 2 : 1;
  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&thread_data[t], (void *)&thread_data[t + tc],
                          xc_config->thread_info.thread_ids.id[t]);
  }
  thread_call((void *)&shared_data, &thread_data[tc - 1], &thread_data[tc - 1 + tc],
              (thread_function_pointer_t)blob_unaryi16_thread_worker,
              &xc_config->thread_info);
  
  return kTfLiteOk;
}

} // namespace blob_unaryi16

TFLMRegistration *Register_XC_blob_unaryi16() {
  static TFLMRegistration r = {blob_unaryi16::Init, nullptr, blob_unaryi16::Prepare,
                               blob_unaryi16::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
