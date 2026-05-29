// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "xcore_custom_options.h"
#include "../thread_call.h"
#include "xcore_config.h"
#include "xcore_utils.h"
#include "lib_nn/api/vpu.hpp"
extern "C" {
#include "lib_nn/api/nn_operator.h"
#include "lib_nn/api/nn_layers.h"
}

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace batch_matmul {

struct BatchMatMulWorkerArg0 {
  int lhs_offset;
  int rhs_offset;
  int output_offset;
  int batch_size;
  int8_t vpu_buf0[VPU_INT8_EPV*2];
  int8_t vpu_buf1[VPU_INT8_EPV*2];
  int8_t vpu_buf2[VPU_INT8_EPV*2];
};

// This is the struct that contains the data required by the operator
struct BatchMatMulOpData {
  int batch_size;
  nn_mat_mul_real_params_t mat_mul_params;
  int tc;
  BatchMatMulWorkerArg0 arg0[XCORE_MAX_NUM_THREADS];
};

struct BatchMatMulShared {
  int8_t *x;
  int8_t *y;
  int8_t *output;
  BatchMatMulOpData *op_data;
};

extern "C" {
void batch_matmul_thread_worker(void *shared, void *arg0, void *arg1) {
  BatchMatMulWorkerArg0 *arg = static_cast<BatchMatMulWorkerArg0 *>(arg0);
  (void) arg1;
  auto sd = static_cast<BatchMatMulShared *>(shared);
  auto op_data = sd->op_data;
  int8_t *lhs = &sd->x[arg->lhs_offset];
  int8_t *rhs = &sd->y[arg->rhs_offset];
  int8_t *output = &sd->output[arg->output_offset];
  int lhs_batch_offset =
    op_data->mat_mul_params.lhs_row_size * op_data->mat_mul_params.channel_size;
  int rhs_batch_offset =
    op_data->mat_mul_params.channel_size * op_data->mat_mul_params.rhs_col_size;
  int output_batch_offset =
    op_data->mat_mul_params.lhs_row_size * op_data->mat_mul_params.rhs_col_size;
  for (int b = 0; b < arg->batch_size; ++b) {
    mat_mul_real_int8(
      &op_data->mat_mul_params,
      arg->vpu_buf0, arg->vpu_buf1, arg->vpu_buf2,
      lhs + b * lhs_batch_offset, rhs + b * rhs_batch_offset,
      output + b * output_batch_offset);
  }
}
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<BatchMatMulOpData>(context);

  auto parser = CustomOptionParser(buffer, length);
  auto compute_shape = parser.parseNamedCustomOption("compute_shape").AsVector();
  op_data->batch_size = compute_shape[0].AsInt32();
  op_data->mat_mul_params.lhs_row_size = compute_shape[1].AsInt32();
  op_data->mat_mul_params.channel_size = compute_shape[2].AsInt32();
  op_data->mat_mul_params.rhs_col_size = compute_shape[3].AsInt32();
  op_data->mat_mul_params.lhs_zp = parser.parseNamedCustomOption("lhs_zp").AsFloat();
  op_data->mat_mul_params.rhs_zp = parser.parseNamedCustomOption("rhs_zp").AsFloat();
  op_data->mat_mul_params.out_zp = parser.parseNamedCustomOption("out_zp").AsFloat();
  op_data->mat_mul_params.scale = parser.parseNamedCustomOption("scale").AsFloat();
  op_data->mat_mul_params.in_zp_sum =
    op_data->mat_mul_params.channel_size *
    op_data->mat_mul_params.lhs_zp *
    op_data->mat_mul_params.rhs_zp;  

  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<BatchMatMulOpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  int s[XCORE_MAX_NUM_THREADS];
  int e[XCORE_MAX_NUM_THREADS];
  op_data->tc = calculateAlignedThreadSplit(
    xc_config->model_thread_count, op_data->batch_size, s, e);
  // Turn start and end into input and output offset
  for (int t = 0; t < op_data->tc; ++t) {
    op_data->arg0[t].lhs_offset =
      s[t] * op_data->mat_mul_params.lhs_row_size * op_data->mat_mul_params.channel_size;
    op_data->arg0[t].rhs_offset =
      s[t] * op_data->mat_mul_params.channel_size * op_data->mat_mul_params.rhs_col_size;
    op_data->arg0[t].output_offset =
      s[t] * op_data->mat_mul_params.lhs_row_size * op_data->mat_mul_params.rhs_col_size;
    op_data->arg0[t].batch_size = e[t]-s[t];
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<BatchMatMulOpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *x = tflite_micro::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *y = tflite_micro::micro::GetEvalInput(context, node, 1);

  TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in X/Y/Out Tensors
  int8_t *out_data = tflite_micro::micro::GetTensorData<int8_t>(output);
  const int8_t *x_data = tflite_micro::micro::GetTensorData<int8_t>(x);
  const int8_t *y_data = tflite_micro::micro::GetTensorData<int8_t>(y);

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());

  BatchMatMulShared shared_data;
  shared_data.x = const_cast<int8_t *>(x_data);
  shared_data.y = const_cast<int8_t *>(y_data);
  shared_data.output = out_data;
  shared_data.op_data = op_data;

  const int tc = op_data->tc;
  if (tc == 1 && x->type == kTfLiteInt8) {
    batch_matmul_thread_worker(&shared_data, &(op_data->arg0[0]), nullptr);
    return kTfLiteOk;
  }

  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&op_data->arg0[t], nullptr,
                          xc_config->thread_info.thread_ids.id[t]);
  }

  thread_call((void *)&shared_data, &op_data->arg0[tc - 1], nullptr,
              (thread_function_pointer_t)batch_matmul_thread_worker, &xc_config->thread_info);

  return kTfLiteOk;
}

} // namespace batch_matmul

TFLMRegistration *Register_XC_batch_matmul() {
  static TFLMRegistration r = {batch_matmul::Init, nullptr, batch_matmul::Prepare, batch_matmul::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
