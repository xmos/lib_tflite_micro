#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "xcore_custom_options.h"
#include "xcore_dispatcher.h"
#include "xcore_utils.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace fully_connected {

struct FullyConnectedOpData {
  ExecutionPlan execution_plan;
  int stack_scratch_index;
  size_t stack_size;
  int weights_scratch_index;
  int bias_scratch_index;
};

struct FullyConnectedThreadData {
  int8_t *Y;
  const nn_tensor_t *X;
  const nn_tensor_t *W;
  const nn_bso_block_t *BSO;
  channel_count_t C_in;
  channel_count_t C_out_start;
  channel_count_t C_out_end;
};

extern "C" {
ATTRIBUTE_THREAD_FUNCTION void fully_connected_thread_worker(void *context) {
  FullyConnectedThreadData *td = (FullyConnectedThreadData *)context;
  fully_connected_8(td->Y, td->W, td->X, td->BSO, td->C_in, td->C_out_start,
                    td->C_out_end);
}
}

void *Init_8(TfLiteContext *context, const char *buffer, size_t length) {
  FullyConnectedOpData *op = nullptr;
  op = reinterpret_cast<FullyConnectedOpData *>(
      context->AllocatePersistentBuffer(context, sizeof(FullyConnectedOpData)));
  op->stack_scratch_index = -1;
  op->stack_size = 0;
  op->weights_scratch_index = -1;
  op->bias_scratch_index = -1;

  TFLITE_DCHECK(buffer != nullptr);
  parse_custom_options(context, buffer, length, &op->execution_plan);

  return op;
}

TfLiteStatus Prepare_8(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor *weights = GetInput(context, node, 1);
  const TfLiteTensor *bso = GetInput(context, node, 2);

  FullyConnectedOpData *op =
      reinterpret_cast<FullyConnectedOpData *>(node->user_data);

  // allocate the stack for thread workers
  GET_THREAD_FUNCTION_STACKSIZE(op->stack_size, fully_connected_thread_worker);
  TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
      context, op->stack_size * op->execution_plan.GetNumThreads(),
      &op->stack_scratch_index));

  // allocate scratch buffers for weights and biases (if necessary)
  if (!is_ram_address((uintptr_t)weights->data.int8)) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, op->execution_plan.GetWeightsScratchSize(),
        &op->weights_scratch_index));
  }
  if (!is_ram_address((uintptr_t)bso->data.i16)) {
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, op->execution_plan.GetBiasScratchSize(),
        &op->bias_scratch_index));
  }

  return kTfLiteOk;
}

TfLiteStatus Eval_8(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  const TfLiteEvalTensor *weights =
      tflite::micro::GetEvalInput(context, node, 1);
  const TfLiteEvalTensor *bso = tflite::micro::GetEvalInput(context, node, 2);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  const RuntimeShape weights_shape = tflite::micro::GetTensorShape(weights);

  FullyConnectedOpData *op =
      reinterpret_cast<FullyConnectedOpData *>(node->user_data);

  int32_t C_in = weights_shape.Dims(1);

  Dispatcher *dispatcher = GetDispatcher();

  // initialize the dispatcher
  char *stack = static_cast<char *>(
      context->GetScratchBuffer(context, op->stack_scratch_index));
  TFLITE_DCHECK(stack != nullptr);
  dispatcher->InitializeTasks(fully_connected_thread_worker, stack,
                              op->stack_size);

  // create thread data and tasks
  int i_th = 0;
  int n_th = op->execution_plan.GetNumThreads();
  FullyConnectedThreadData thread_data[n_th];

  // load weights & bias scratch buffers(if necessary)
  size_t weights_dest_offset = 0;
  size_t weights_src_offset = 0;
  size_t biases_dest_offset = 0;
  size_t biases_src_offset = 0;
  size_t weights_fetch_size;
  // size_t bias_fetch_size;
  int8_t *sW, *tW; // sW points to the head of the weights scratch space, tW
                   // points to the head of the fetched weights which equals sW
                   // for the first fetch but not for subsequent fetches
  int8_t *sBSO,
      *tBSO; // sBSO points to the head of the BSO scratch space, tBSO
             // points to the head of the fetched BSO which equals sBSO for
             // the first fetch but not for subsequent fetches

  if (op->weights_scratch_index >= 0) {
    sW = static_cast<int8_t *>(
        context->GetScratchBuffer(context, op->weights_scratch_index));
    TFLITE_DCHECK(sW != nullptr);
  }
  if (op->bias_scratch_index >= 0) {
    sBSO = static_cast<int8_t *>(
        context->GetScratchBuffer(context, op->bias_scratch_index));
    TFLITE_DCHECK(sBSO != nullptr);
  }

  for (int i_cg = 0; i_cg < op->execution_plan.changrps.size(); i_cg++) {
    const ChannelGroup &changrp = op->execution_plan.changrps[i_cg];

    // offset into the temp W and BSO pointers based on how many bytes we
    // have loaded since the last JoinTasks
    tW = sW + weights_dest_offset;
    tBSO = sBSO + biases_dest_offset;

    // fetch the weights and biases
    weights_fetch_size = C_in * changrp.size;
    FetchBuffer(
        &tW, &tflite::micro::GetTensorData<int8_t>(weights)[weights_src_offset],
        weights_fetch_size);
    weights_dest_offset += weights_fetch_size;
    weights_src_offset += weights_fetch_size;

    FetchBuffer((int8_t **)&tBSO,
                &tflite::micro::GetTensorData<int8_t>(bso)[biases_src_offset],
                kBSOChannelGroupBytes);
    biases_dest_offset += kBSOChannelGroupBytes;
    biases_src_offset += kBSOChannelGroupBytes;

    thread_data[i_th].Y =
        &tflite::micro::GetTensorData<int8_t>(output)[changrp.start];
    thread_data[i_th].X = tflite::micro::GetTensorData<int8_t>(input);
    thread_data[i_th].W = tW;
    thread_data[i_th].BSO = (const nn_bso_block_t *)tBSO;
    thread_data[i_th].C_in = C_in;
    thread_data[i_th].C_out_start = 0;
    thread_data[i_th].C_out_end = thread_data[i_th].C_out_start + changrp.size;
    dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[i_th]));

    i_th++;

    if (i_th == n_th) {
      dispatcher->JoinTasks();
      i_th = 0;
      weights_dest_offset = 0;
      biases_dest_offset = 0;
    }
  }
  dispatcher->JoinTasks(); // finish up any added tasks

  return kTfLiteOk;
}

} // namespace fully_connected

TfLiteRegistration *Register_FullyConnected_8() {
  static TfLiteRegistration r = {fully_connected::Init_8, nullptr,
                                 fully_connected::Prepare_8,
                                 fully_connected::Eval_8};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
