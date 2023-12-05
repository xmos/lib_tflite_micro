// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "lib_nn/api/AbstractKernel.hpp"
#include "lib_nn/api/AggregateFn.hpp"
#include "lib_nn/api/MemCpyFn.hpp"
#include "lib_nn/api/OutputTransformFn.hpp"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace maxpool_2d {

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //

struct MaxPool2DShared {
  int8_t *X;
  int8_t *Y;
  nn::conv_params_t *conv_params;
};

extern "C" {
// TODO
#pragma stackfunction 1000
void maxpool2d_thread_worker(void *shard, void *scrtch, void *kp) {
  nn::abstract_kernel_params_t *akparams = (nn::abstract_kernel_params_t *)kp;
  auto scratch = static_cast<int8_t *>(scrtch);
  auto shared = static_cast<MaxPool2DShared *>(shard);
  execute(shared->Y, shared->X, shared->conv_params, akparams, NULL, NULL,
          false, scratch);
}
}

/**
 * @brief This describes the memory requirements of a worker thread. It also
 * includes an array of the work to be done by said worker.
 *
 */
struct MaxPool2DThreadInfo {
  size_t scratch_size;     // Each thread needs a scratch
  int stack_scratch_index; // All threads stack and scratch consolidated into a
                           // single scratch buffer
  nn::abstract_kernel_params_t *kparams;
};

// This is the struct that contains the data required to fully describe the work
// that the operator will perform. It needs to descibe the work for T threads.
// That means it must contain:
// - T sets of work, i.e. a list of jobs for each thread.
// - T scratch allocations, i.e. an amount of scratch memory for each thread.
struct MaxPool2DOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  size_t thread_count;
  MaxPool2DThreadInfo *threads;
  nn::conv_params_t maxpool_params; // The job to be done by this thread
};

// -------------------------------------------------------------------- //
// op function implementations
// -------------------------------------------------------------------- //

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  TFLITE_DCHECK(buffer != nullptr);
  auto op_data = construct_persistent_object<MaxPool2DOpData>(context);
  auto parser = CustomOptionParser(buffer, length);

  const uint8_t *memcpy_fn_data =
      parser.parseNamedCustomOption("mp").AsBlob().data();
  const uint8_t *agg_fn_data =
      parser.parseNamedCustomOption("a").AsBlob().data();
  const uint8_t *ot_fn_data =
      parser.parseNamedCustomOption("o").AsBlob().data();
  int32_t scratch_size = parser.parseNamedCustomOption("s").AsInt32();
  int32_t ot_type = parser.parseNamedCustomOption("t").AsInt32();
  auto ak_params_vec = parser.parseNamedCustomOption("p").AsVector();

  // Unlike other ops, conv ops have their own individual thread count
  // The work for each thread is calculated in the compiler
  auto thread_count = ak_params_vec.size();
  op_data->thread_count = thread_count;
  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());
  assert(op_data->thread_count <= xc_config->model_thread_count &&
         "Not enough threads!");

  op_data->maxpool_params.mem_p = (nn::memcpyfn_deref_params_t *)memcpy_fn_data;
  op_data->maxpool_params.agg_p = (nn::mat_mul_dw_direct_params_t *)agg_fn_data;
  op_data->maxpool_params.ot_p =
      (nn::otfn_int8_channelwise_params_t *)ot_fn_data;
  op_data->threads =
      static_cast<MaxPool2DThreadInfo *>(context->AllocatePersistentBuffer(
          context, op_data->thread_count * sizeof(MaxPool2DThreadInfo)));
  op_data->maxpool_params.memcopy_fn = (nn::MemFnType)nn::memcpyfn_deref;
  op_data->maxpool_params.aggregate_fn = (nn::AggFnType)nn::maxpool_direct;
  op_data->maxpool_params.output_transform_fn =
      (nn::OtFnType)nn::otfn_int8_maxpool;
  // TODO:
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = reinterpret_cast<MaxPool2DOpData *>(node->user_data);
  for (int t = 0; t < op_data->thread_count; ++t) {
    if (op_data->threads[t].scratch_size) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, op_data->threads[t].scratch_size,
          &op_data->threads[t].stack_scratch_index));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  const TfLiteEvalTensor *multipliers_and_biases_tensor =
      tflite::micro::GetEvalInput(context, node, 2);

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());

  auto *op_data = reinterpret_cast<MaxPool2DOpData *>(node->user_data);
  int n_threads = op_data->thread_count;

  int8_t *thread_scratch[XCORE_MAX_NUM_THREADS];
  MaxPool2DShared shared_data;
  shared_data.X = (int8_t *)tflite::micro::GetTensorData<int8_t>(input);
  shared_data.Y = (int8_t *)tflite::micro::GetTensorData<int8_t>(output);
  shared_data.conv_params = &op_data->maxpool_params;
  for (int t = 0; t < n_threads; ++t) {
    if (op_data->threads[t].scratch_size) {
      thread_scratch[t] = (int8_t *)context->GetScratchBuffer(
          context, op_data->threads[t].stack_scratch_index);
    }
  }

  // todo - this second for-loop is unpleasant
  for (int t = 0; t < n_threads - 1; ++t) {
    thread_variable_setup(thread_scratch[t], op_data->threads[t].kparams,
                          xc_config->thread_info.thread_ids.id[t]);
  }

  // Now set up shared data, shared function pointer, and data for final thread.
  thread_call((void *)&shared_data, thread_scratch[n_threads - 1],
              op_data->threads[n_threads - 1].kparams,
              (thread_function_pointer_t)maxpool2d_thread_worker,
              &xc_config->thread_info);

  return kTfLiteOk;
}

} // namespace maxpool_2d

TFLMRegistration *Register_XC_maxpool2d() {
  static TFLMRegistration r = {maxpool_2d::Init, nullptr, maxpool_2d::Prepare,
                               maxpool_2d::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
