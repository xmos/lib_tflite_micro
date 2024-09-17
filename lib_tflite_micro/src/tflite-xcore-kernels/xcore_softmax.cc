#include "../thread_call.h"
#include "xcore_common.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace softmax {

struct SoftmaxShared {
  int8_t *X, *Y;
  float *table;
  float inv_sum;
};

struct SoftmaxIdx {
  int start, end;
};

extern "C" {
void exp_sum_thread_worker(void *shared, void *idx, void *sum_ptr) {
  const auto sidx = static_cast<SoftmaxIdx *>(idx);
  const unsigned s = sidx->start;
  const unsigned e = sidx->end;
  float *sum = static_cast<float *>(sum_ptr);
  auto sd = static_cast<SoftmaxShared *>(shared);
  softmax_exp_sum(sum, sd->X, sd->table, s, e - s);
}

void exp_div_thread_worker(void *shared, void *idx, void *useless) {
  const auto sidx = static_cast<SoftmaxIdx *>(idx);
  const unsigned s = sidx->start;
  const unsigned e = sidx->end;
  auto sd = static_cast<SoftmaxShared *>(shared);
  const float inv_sum_f = static_cast<float>(sd->inv_sum);
  softmax_exp_div(sd->Y, sd->X, sd->table, inv_sum_f, s, e - s);
}
}

// This is the struct that contains the data required by the operator
struct SoftmaxOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int tc;
  SoftmaxIdx idx[XCORE_MAX_NUM_THREADS];
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<SoftmaxOpData>(context);
  op_data->name = "XC_softmax";
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<SoftmaxOpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const TfLiteEvalTensor *input = tflite_micro::micro::GetEvalInput(context, node, 0);
  int input_size = tflite_micro::micro::GetTensorShape(input).FlatSize();
  op_data->tc = xc_config->model_thread_count;
  int s[XCORE_MAX_NUM_THREADS];
  int e[XCORE_MAX_NUM_THREADS];
  calculateThreadSplit(op_data->tc, input_size, s, e);
  for (int t = 0; t < op_data->tc; t++) {
    op_data->idx[t] = {s[t], e[t]};
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<SoftmaxOpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite_micro::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *table = tflite_micro::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  const float *table_vals = tflite_micro::micro::GetTensorData<float>(table);
  int8_t *out_data = tflite_micro::micro::GetTensorData<int8_t>(output);
  const int8_t *in_data = tflite_micro::micro::GetTensorData<int8_t>(input);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const int tc = op_data->tc;
  float sums[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  SoftmaxShared shared_data;
  shared_data.Y = out_data;
  shared_data.X = const_cast<int8_t *>(in_data);
  shared_data.table = const_cast<float *>(table_vals);
  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&op_data->idx[t], (void *)&sums[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }
  thread_call((void *)&shared_data, (void *)&op_data->idx[tc - 1],
              (void *)&sums[tc - 1],
              (thread_function_pointer_t)exp_sum_thread_worker,
              &xc_config->thread_info);
  softmax_calculate_inv_sum(&shared_data.inv_sum, sums);

  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&op_data->idx[t], (void *)&sums[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }
  thread_call((void *)&shared_data, (void *)&op_data->idx[tc - 1],
              (void *)&sums[tc - 1],
              (thread_function_pointer_t)exp_div_thread_worker,
              &xc_config->thread_info);
  return kTfLiteOk;
}
} // namespace softmax

TFLMRegistration *Register_XC_softmax() {
  static TFLMRegistration r = {softmax::Init, nullptr, softmax::Prepare,
                               softmax::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
