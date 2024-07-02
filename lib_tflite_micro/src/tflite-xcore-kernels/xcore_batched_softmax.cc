
#include "../thread_call.h"
#include "xcore_common.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace softmax_batched {

struct SoftmaxBatchedShared {
  int8_t *X, *Y;
  int softmax_size;
  float *table;
};

extern "C" {
void softmax_thread_worker(void *shared, void *start, void *count) {
  int *s = static_cast<int *>(start);
  int *c = static_cast<int *>(count);
  auto sd = static_cast<SoftmaxBatchedShared *>(shared);
  for (int i = 0; i < *c; i++) {
    const int offset = i * sd->softmax_size + *s;
    softmax_single(sd->Y + offset, sd->X + offset, sd->table, sd->softmax_size);
  }
}
}

// This is the struct that contains the data required by the operator
struct SoftmaxBatchedOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  int tc;
  int softmax_size;
  int starts[XCORE_MAX_NUM_THREADS];
  int counts[XCORE_MAX_NUM_THREADS];
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<SoftmaxBatchedOpData>(context);
  op_data->name = "XC_softmax_batched";
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto op_data = static_cast<SoftmaxBatchedOpData *>(node->user_data);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  const int num_softmaxes = tflite::micro::GetTensorShape(input).Dims(0);
  op_data->softmax_size = tflite::micro::GetTensorShape(input).Dims(1);
  op_data->tc = xc_config->model_thread_count;
  int starts[XCORE_MAX_NUM_THREADS];
  int ends[XCORE_MAX_NUM_THREADS];
  int counts[XCORE_MAX_NUM_THREADS];
  calculateThreadSplit(op_data->tc, num_softmaxes, starts, ends);
  for (int t = 0; t < op_data->tc; t++) {
    op_data->counts[t] = ends[t] - starts[t];
    op_data->starts[t] = starts[t] * op_data->softmax_size;
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {

  auto *op_data = static_cast<SoftmaxBatchedOpData *>(node->user_data);

  // Get Input/Output Tensors
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *table = tflite::micro::GetEvalInput(context, node, 1);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  // Pointers to data in In/Out Tensors
  const float *table_vals = tflite::micro::GetTensorData<float>(table);
  int8_t *out_data = tflite::micro::GetTensorData<int8_t>(output);
  const int8_t *in_data = tflite::micro::GetTensorData<int8_t>(input);
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  const int tc = op_data->tc;
  SoftmaxBatchedShared shared_data;
  shared_data.Y = out_data;
  shared_data.X = const_cast<int8_t *>(in_data);
  shared_data.table = const_cast<float *>(table_vals);
  shared_data.softmax_size = op_data->softmax_size;
  for (int t = 0; t < tc - 1; t++) {
    thread_variable_setup((void *)&op_data->starts[t],
                          (void *)&op_data->counts[t],
                          xc_config->thread_info.thread_ids.id[t]);
  }
  thread_call((void *)&shared_data, (void *)&op_data->starts[tc - 1],
              (void *)&op_data->counts[tc - 1],
              (thread_function_pointer_t)softmax_thread_worker,
              &xc_config->thread_info);
  return kTfLiteOk;
}
} // namespace softmax_batched

TFLMRegistration *Register_XC_batched_softmax() {
  static TFLMRegistration r = {softmax_batched::Init, nullptr,
                               softmax_batched::Prepare, softmax_batched::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
