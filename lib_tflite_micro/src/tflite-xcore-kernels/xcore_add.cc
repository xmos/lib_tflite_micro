#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "xcore_dispatcher.h"
#include "xcore_elementwise.h"
#include "xcore_utils.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace add {

// -------------------------------------------------------------------- //
// kernel argument type
// -------------------------------------------------------------------- //

struct AddArguments {
  int8_t *Y;
  const int8_t *X0;
  const int8_t *X1;
  nn_add_params_t params;
};

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //
using AddThreadData = ElementwiseThreadData<AddArguments>;

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void add_thread_worker(void *context) {
  auto *td = static_cast<AddThreadData *>(context);
  auto *args = td->args;
  add_elementwise(args->Y, args->X0, args->X1, &args->params, td->start,
                  td->element_count);
}
}

// -------------------------------------------------------------------- //
// op data types
// -------------------------------------------------------------------- //

struct AddOpData : MultiThreadedOpData<AddArguments, AddThreadData> {
  // TODO: remove this when better external memory handling is implemented
  // for loading from external mem
  int input0_scratch_idx = -1;
  int input1_scratch_idx = -1;
};

// -------------------------------------------------------------------- //
// op function implementations
// -------------------------------------------------------------------- //

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  auto *op_data = reinterpret_cast<AddOpData *>(node->user_data);

  // TODO: memory map this instead
  const auto *bss = GetInput(context, node, 2);
  auto &params = op_data->args.params;
  params.input[0].shr = bss->data.i32[0];
  params.input[0].multiplier = bss->data.i32[1];
  params.input[1].shr = bss->data.i32[2];
  params.input[1].multiplier = bss->data.i32[3];
  params.output.bias = bss->data.i32[4];
  params.output.shr = bss->data.i32[5];

  // allocate the stack for thread workers
  GET_THREAD_FUNCTION_STACKSIZE(op_data->stack_size, add_thread_worker);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op_data->stack_size * op_data->threads.size(),
      &op_data->stack_scratch_index));

  // TODO: remove this when better fetching is implemented
  TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
      context, GetInput(context, node, 0), op_data->input0_scratch_idx));
  TF_LITE_ENSURE_STATUS(request_scratch_if_needed(
      context, GetInput(context, node, 1), op_data->input1_scratch_idx));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = reinterpret_cast<AddOpData *>(node->user_data);

  TF_LITE_ENSURE_STATUS(fetch_scratch_if_needed(
      context, op_data->args.X0, tflite::micro::GetEvalInput(context, node, 0),
      op_data->input0_scratch_idx));
  TF_LITE_ENSURE_STATUS(fetch_scratch_if_needed(
      context, op_data->args.X1, tflite::micro::GetEvalInput(context, node, 1),
      op_data->input1_scratch_idx));
  op_data->args.Y = tflite::micro::GetTensorData<int8_t>(
      tflite::micro::GetEvalOutput(context, node, 0));

  // initialize the dispatcher
  Dispatcher *dispatcher = GetDispatcher();
  auto *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op_data->stack_scratch_index));
  TF_LITE_ENSURE(context, stack);
  dispatcher->InitializeTasks(add_thread_worker, stack, op_data->stack_size);

  for (auto &thread : op_data->threads) {
    dispatcher->AddTask(reinterpret_cast<void *>(&thread));
  }
  dispatcher->JoinTasks();

  return kTfLiteOk;
}

} // namespace add

TfLiteRegistration *Register_Add_8() {
  static TfLiteRegistration r = {ElementwiseInit<add::AddOpData>, nullptr,
                                 add::Prepare, add::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
