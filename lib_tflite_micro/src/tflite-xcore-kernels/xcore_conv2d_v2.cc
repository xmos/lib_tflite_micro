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
#include "xcore_dispatcher.h"
#include "xcore_interpreter.h"
#include "xcore_utils.h"
extern "C" {
#include "nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace conv_v2 {

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //

struct Conv2DThread {
  int8_t *X;
  int8_t *Y;
  int8_t *scratch;

  // TODO: Clean up
  // Using AbstractKernel to be able to assign Filter2D or Filter2D_DW
  nn::AbstractKernel *f;
};

extern "C" {
// TODO
#pragma stackfunction 1000
ATTRIBUTE_THREAD_FUNCTION
void conv2d_v2_thread_worker(void *context) {
  auto work = static_cast<Conv2DThread *>(context);
  work->f->execute(work->Y, work->X, work->scratch);
}
}

// -------------------------------------------------------------------- //
// op data types
// -------------------------------------------------------------------- //

enum KernelType {
  Conv2DValidDirect_t,
  Conv2DValidIndirect_t,
  Conv2DPaddedIndirect_t,
  DepthwiseConv2DValidDirect_t,
  DepthwiseConv2DPaddedIndirect_t,
  BNNConv2DValidDirectBinary_t,
  BNNConv2DValidIndirectBinary_t,
  BNNConv2DValidDirectInt8_t,
  BNNConv2DValidIndirectInt8_t
};

/**
 * @brief This describes the memory requirements of a worker thread. It also
 * includes an array of the work to be done by said worker.
 *
 */
struct Conv2DThreadInfo {
  size_t stack_size;       // Each thread needs a stack
  size_t scratch_size;     // Each thread needs a scratch
  int stack_scratch_index; // All threads stack and scratch consolidated into a
                           // single scratch buffer
  // TODO: Clean up
  // Using AbstractKernel to be able to assign Filter2D or Filter2D_DW
  nn::AbstractKernel *filter2D; // The job to be done by this thread
  KernelType kt;
};

// This is the struct that contains the data required to fully describe the work
// that the operator will perform. It needs to descibe the work for T threads.
// That means it must contain:
// - T sets of work, i.e. a list of jobs for each thread.
// - T scratch allocations, i.e. an amount of scratch memory for each thread.
struct Conv2DOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  size_t thread_count;
  Conv2DThreadInfo *threads;
};

// -------------------------------------------------------------------- //
// op function implementations
// -------------------------------------------------------------------- //

template <typename T>
T *getDeserializedParams(TfLiteContext *context, const uint8_t *data) {
  char *allocated_memory;
  int allocationByteCount = sizeof(T);
  allocated_memory =
      (char *)context->AllocatePersistentBuffer(context, allocationByteCount);
  T *param = T::template deserialise<T>(allocated_memory, (const char *)data);
  return param;
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  TFLITE_DCHECK(buffer != nullptr);

  auto op_data = construct_persistent_object<Conv2DOpData>(context);
  auto parser = CustomOptionParser(buffer, length);
  auto threads = parser.parseNamedCustomOption("threads").AsVector();
  auto thread_count = threads.size();
  op_data->thread_count = thread_count;
  op_data->threads =
      static_cast<Conv2DThreadInfo *>(context->AllocatePersistentBuffer(
          context, op_data->thread_count * sizeof(Conv2DThreadInfo)));

  for (int t = 0; t < op_data->thread_count; ++t) {
    flexbuffers::Vector params = threads[t].AsVector();
    op_data->threads[t].scratch_size = params[0].AsInt32();
    // read the kernel type
    KernelType kt = (KernelType)params[1].AsInt32();
    op_data->threads[t].kt = kt;

    switch (kt) {
    // TODO : Cleanup to combine
    case Conv2DValidDirect_t: {
      nn::Filter2D::Params *ak_params =
          getDeserializedParams<nn::Filter2D::Params>(
              context, params[2].AsBlob().data());
      nn::DerefInputFn::Params *mf_params =
          getDeserializedParams<nn::DerefInputFn::Params>(
              context, params[3].AsBlob().data());
      nn::MatMulDirectFn::Params *af_params =
          getDeserializedParams<nn::MatMulDirectFn::Params>(
              context, params[4].AsBlob().data());
      nn::OT_int8::Params *ot_params =
          getDeserializedParams<nn::OT_int8::Params>(context,
                                                     params[5].AsBlob().data());

      // TODO: Make part of construct_persistent_object
      auto memcpy = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::DerefInputFn))) nn::DerefInputFn(mf_params);

      auto aggregator = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::MatMulDirectFn))) nn::MatMulDirectFn(af_params);

      auto ot =
          new (context->AllocatePersistentBuffer(context, sizeof(nn::OT_int8)))
              nn::OT_int8(ot_params);

      auto conv2d = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::Conv2dValidDirect)))
          nn::Conv2dValidDirect(ak_params, memcpy, aggregator, ot);

      op_data->threads[t].filter2D = conv2d;
      op_data->name = "XC_Conv2DValidDir";
    } break;
    case Conv2DValidIndirect_t: {
      nn::Filter2D::Params *ak_params =
          getDeserializedParams<nn::Filter2D::Params>(
              context, params[2].AsBlob().data());
      nn::ImToColValid::Params *mf_params =
          getDeserializedParams<nn::ImToColValid::Params>(
              context, params[3].AsBlob().data());
      nn::MatMulInt8::Params *af_params =
          getDeserializedParams<nn::MatMulInt8::Params>(
              context, params[4].AsBlob().data());
      nn::OT_int8::Params *ot_params =
          getDeserializedParams<nn::OT_int8::Params>(context,
                                                     params[5].AsBlob().data());

      auto memcpy = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::ImToColValid))) nn::ImToColValid(mf_params);

      auto aggregator = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::MatMulInt8))) nn::MatMulInt8(af_params);

      auto ot =
          new (context->AllocatePersistentBuffer(context, sizeof(nn::OT_int8)))
              nn::OT_int8(ot_params);

      auto conv2d = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::Conv2dValidIndirect)))
          nn::Conv2dValidIndirect(ak_params, memcpy, aggregator, ot);

      op_data->threads[t].filter2D = conv2d;
      op_data->name = "XC_Conv2DValidInd";
    } break;
    case Conv2DPaddedIndirect_t: {
      nn::Filter2D::Params *ak_params =
          getDeserializedParams<nn::Filter2D::Params>(
              context, params[2].AsBlob().data());
      nn::ImToColPadded::Params *mf_params =
          getDeserializedParams<nn::ImToColPadded::Params>(
              context, params[3].AsBlob().data());
      nn::MatMulInt8::Params *af_params =
          getDeserializedParams<nn::MatMulInt8::Params>(
              context, params[4].AsBlob().data());
      nn::OT_int8::Params *ot_params =
          getDeserializedParams<nn::OT_int8::Params>(context,
                                                     params[5].AsBlob().data());

      auto memcpy = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::ImToColPadded))) nn::ImToColPadded(mf_params);

      auto aggregator = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::MatMulInt8))) nn::MatMulInt8(af_params);

      auto ot =
          new (context->AllocatePersistentBuffer(context, sizeof(nn::OT_int8)))
              nn::OT_int8(ot_params);

      auto conv2d = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::Conv2dPaddedInDirect)))
          nn::Conv2dPaddedInDirect(ak_params, memcpy, aggregator, ot);

      op_data->threads[t].filter2D = conv2d;
      op_data->name = "XC_Conv2DPadInd";
    } break;
    case DepthwiseConv2DValidDirect_t: {
      nn::Filter2D_DW::Params *ak_params =
          getDeserializedParams<nn::Filter2D_DW::Params>(
              context, params[2].AsBlob().data());
      nn::DerefInputFn::Params *mf_params =
          getDeserializedParams<nn::DerefInputFn::Params>(
              context, params[3].AsBlob().data());
      nn::MatMulDirectFn_DW::Params *af_params =
          getDeserializedParams<nn::MatMulDirectFn_DW::Params>(
              context, params[4].AsBlob().data());
      nn::OT_int8::Params *ot_params =
          getDeserializedParams<nn::OT_int8::Params>(context,
                                                     params[5].AsBlob().data());

      auto memcpy = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::DerefInputFn))) nn::DerefInputFn(mf_params);

      auto aggregator = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::MatMulDirectFn_DW)))
          nn::MatMulDirectFn_DW(af_params);

      auto ot =
          new (context->AllocatePersistentBuffer(context, sizeof(nn::OT_int8)))
              nn::OT_int8(ot_params);

      auto conv2d = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::Conv2dDepthwiseValidDirect)))
          nn::Conv2dDepthwiseValidDirect(ak_params, memcpy, aggregator, ot);

      op_data->threads[t].filter2D = conv2d;
      op_data->name = "XC_DWConv2DValidInd";
    } break;
    case DepthwiseConv2DPaddedIndirect_t: {
      nn::Filter2D_DW::Params *ak_params =
          getDeserializedParams<nn::Filter2D_DW::Params>(
              context, params[2].AsBlob().data());
      nn::ImToColPadded::Params *mf_params =
          getDeserializedParams<nn::ImToColPadded::Params>(
              context, params[3].AsBlob().data());
      nn::MatMulDirectFn_DW::Params *af_params =
          getDeserializedParams<nn::MatMulDirectFn_DW::Params>(
              context, params[4].AsBlob().data());
      nn::OT_int8::Params *ot_params =
          getDeserializedParams<nn::OT_int8::Params>(context,
                                                     params[5].AsBlob().data());

      auto memcpy = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::ImToColPadded))) nn::ImToColPadded(mf_params);

      auto aggregator = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::MatMulDirectFn_DW)))
          nn::MatMulDirectFn_DW(af_params);

      auto ot =
          new (context->AllocatePersistentBuffer(context, sizeof(nn::OT_int8)))
              nn::OT_int8(ot_params);

      auto conv2d = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::Conv2dDepthwisePaddedIndirect)))
          nn::Conv2dDepthwisePaddedIndirect(ak_params, memcpy, aggregator, ot);

      op_data->threads[t].filter2D = conv2d;
      op_data->name = "XC_DWConv2DPadInd";
    } break;
    case BNNConv2DValidDirectBinary_t: {
      nn::Filter2D::Params *ak_params =
          getDeserializedParams<nn::Filter2D::Params>(
              context, params[2].AsBlob().data());
      nn::DerefInputFn::Params *mf_params =
          getDeserializedParams<nn::DerefInputFn::Params>(
              context, params[3].AsBlob().data());
      nn::MatMulBinaryDirectFn::Params *af_params =
          getDeserializedParams<nn::MatMulBinaryDirectFn::Params>(
              context, params[4].AsBlob().data());

      // TODO: Make part of construct_persistent_object
      auto memcpy = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::DerefInputFn))) nn::DerefInputFn(mf_params);

      auto aggregator = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::MatMulBinaryDirectFn)))
          nn::MatMulBinaryDirectFn(af_params);

      auto ot = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::OT_binary))) nn::OT_binary();

      auto conv2d = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::BNNConv2dValidDirectBinary)))
          nn::BNNConv2dValidDirectBinary(ak_params, memcpy, aggregator, ot);

      op_data->threads[t].filter2D = conv2d;
      op_data->name = "XC_BNNValidDirBin";
    } break;
    case BNNConv2DValidIndirectBinary_t: {
      nn::Filter2D::Params *ak_params =
          getDeserializedParams<nn::Filter2D::Params>(
              context, params[2].AsBlob().data());
      nn::ImToColValid::Params *mf_params =
          getDeserializedParams<nn::ImToColValid::Params>(
              context, params[3].AsBlob().data());
      nn::MatMulBinary::Params *af_params =
          getDeserializedParams<nn::MatMulBinary::Params>(
              context, params[4].AsBlob().data());

      // TODO: Make part of construct_persistent_object
      auto memcpy = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::ImToColValid))) nn::ImToColValid(mf_params);

      auto aggregator = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::MatMulBinary))) nn::MatMulBinary(af_params);

      auto ot = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::OT_binary))) nn::OT_binary();

      auto conv2d = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::BNNConv2dValidIndirectBinary)))
          nn::BNNConv2dValidIndirectBinary(ak_params, memcpy, aggregator, ot);

      op_data->threads[t].filter2D = conv2d;
      op_data->name = "XC_BNNValidIndBin";
    } break;
    case BNNConv2DValidDirectInt8_t: {
      nn::Filter2D::Params *ak_params =
          getDeserializedParams<nn::Filter2D::Params>(
              context, params[2].AsBlob().data());
      nn::DerefInputFn::Params *mf_params =
          getDeserializedParams<nn::DerefInputFn::Params>(
              context, params[3].AsBlob().data());
      nn::MatMulBinaryDirectFn::Params *af_params =
          getDeserializedParams<nn::MatMulBinaryDirectFn::Params>(
              context, params[4].AsBlob().data());
      nn::OT_int8_clamped::Params *ot_params =
          getDeserializedParams<nn::OT_int8_clamped::Params>(
              context, params[5].AsBlob().data());

      // TODO: Make part of construct_persistent_object
      auto memcpy = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::DerefInputFn))) nn::DerefInputFn(mf_params);

      auto aggregator = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::MatMulBinaryDirectFn)))
          nn::MatMulBinaryDirectFn(af_params);

      auto ot = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::OT_int8_clamped))) nn::OT_int8_clamped(ot_params);

      auto conv2d = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::BNNConv2dValidDirectInt8)))
          nn::BNNConv2dValidDirectInt8(ak_params, memcpy, aggregator, ot);

      op_data->threads[t].filter2D = conv2d;
      op_data->name = "XC_BNNValidDirInt8";
    } break;
    case BNNConv2DValidIndirectInt8_t: {
      nn::Filter2D::Params *ak_params =
          getDeserializedParams<nn::Filter2D::Params>(
              context, params[2].AsBlob().data());
      nn::ImToColValid::Params *mf_params =
          getDeserializedParams<nn::ImToColValid::Params>(
              context, params[3].AsBlob().data());
      nn::MatMulBinary::Params *af_params =
          getDeserializedParams<nn::MatMulBinary::Params>(
              context, params[4].AsBlob().data());
      nn::OT_int8_clamped::Params *ot_params =
          getDeserializedParams<nn::OT_int8_clamped::Params>(
              context, params[5].AsBlob().data());

      // TODO: Make part of construct_persistent_object
      auto memcpy = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::ImToColValid))) nn::ImToColValid(mf_params);

      auto aggregator = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::MatMulBinary))) nn::MatMulBinary(af_params);

      auto ot = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::OT_int8_clamped))) nn::OT_int8_clamped(ot_params);

      auto conv2d = new (context->AllocatePersistentBuffer(
          context, sizeof(nn::BNNConv2dValidIndirectInt8)))
          nn::BNNConv2dValidIndirectInt8(ak_params, memcpy, aggregator, ot);

      op_data->threads[t].filter2D = conv2d;
      op_data->name = "XC_BNNValidIndInt8";
    } break;
    }
  }
  return op_data;
}

// Does all the requests for scratches
TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  // TODO
  // TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  // TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  auto *op_data = reinterpret_cast<Conv2DOpData *>(node->user_data);
  for (int t = 0; t < op_data->thread_count; ++t) {
    // allocate the stack for thread workers
    size_t require_stack;
    // get stack size
    GET_THREAD_FUNCTION_STACKSIZE(require_stack, conv2d_v2_thread_worker);
    op_data->threads[t].stack_size = require_stack;
    size_t request =
        op_data->threads[t].scratch_size + op_data->threads[t].stack_size;
    TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, request, &op_data->threads[t].stack_scratch_index));
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);
  const TfLiteEvalTensor *weights_tensor =
      tflite::micro::GetEvalInput(context, node, 1);
  const TfLiteEvalTensor *multipliers_and_biases_tensor =
      tflite::micro::GetEvalInput(context, node, 2);

  auto *op_data = reinterpret_cast<Conv2DOpData *>(node->user_data);
  int n_threads = op_data->thread_count;
  auto *dispatcher = GetDispatcher();

  for (int t = 0; t < n_threads; ++t) {
    // TODO: Can stack and scratch be the same?
    auto *stack = static_cast<char *>(context->GetScratchBuffer(
        context, op_data->threads[t].stack_scratch_index));
    TF_LITE_ENSURE(context, stack);

    dispatcher->InitializeTasks(conv2d_v2_thread_worker, stack,
                                op_data->threads[t].stack_size);
  }

  // If the no of threads is more than one, we need to split the weights and
  // point to the correct offset for each thread
  assert(n_threads == 1);
  int8_t *weights =
      (int8_t *)tflite::micro::GetTensorData<int8_t>(weights_tensor);
  int16_t *multipliers_and_biases =
      (int16_t *)tflite::micro::GetTensorData<int16_t>(
          multipliers_and_biases_tensor);

  Conv2DThread thread_data[n_threads];
  void *dispatcher_args[n_threads];
  for (int t = 0; t < n_threads; ++t) {
    thread_data[t].X = (int8_t *)tflite::micro::GetTensorData<int8_t>(input);
    thread_data[t].Y = (int8_t *)tflite::micro::GetTensorData<int8_t>(output);
    thread_data[t].scratch = (int8_t *)context->GetScratchBuffer(
        context, op_data->threads[t].stack_scratch_index);

    switch (op_data->threads[t].kt) {
    case Conv2DValidDirect_t: {
      nn::Filter2D *f = (nn::Filter2D *)op_data->threads[t].filter2D;
      nn::MatMulDirectFn *aggr = (nn::MatMulDirectFn *)(f->aggregate_handler);
      aggr->setWeights(weights);
      nn::OT_int8 *ot = (nn::OT_int8 *)(f->ot_handler);
      ot->setMultipliersAndBiases(multipliers_and_biases);
    } break;
    case Conv2DValidIndirect_t:
    case Conv2DPaddedIndirect_t: {
      nn::Filter2D *f = (nn::Filter2D *)op_data->threads[t].filter2D;
      nn::MatMulInt8 *aggr = (nn::MatMulInt8 *)(f->aggregate_handler);
      aggr->setWeights(weights);
      nn::OT_int8 *ot = (nn::OT_int8 *)(f->ot_handler);
      ot->setMultipliersAndBiases(multipliers_and_biases);
    } break;
    case DepthwiseConv2DPaddedIndirect_t:
    case DepthwiseConv2DValidDirect_t: {
      nn::Filter2D_DW *f = (nn::Filter2D_DW *)op_data->threads[t].filter2D;
      nn::MatMulDirectFn_DW *aggr =
          (nn::MatMulDirectFn_DW *)(f->aggregate_handler);
      aggr->setWeights(weights);
      nn::OT_int8 *ot = (nn::OT_int8 *)(f->ot_handler);
      ot->setMultipliersAndBiases(multipliers_and_biases);
    } break;
    case BNNConv2DValidDirectBinary_t: {
      nn::Filter2D *f = (nn::Filter2D *)op_data->threads[t].filter2D;
      nn::MatMulBinaryDirectFn *aggr =
          (nn::MatMulBinaryDirectFn *)(f->aggregate_handler);
      aggr->setWeights(weights);
      nn::OT_binary *ot = (nn::OT_binary *)(f->ot_handler);
      ot->setThresholds(multipliers_and_biases);
    } break;
    case BNNConv2DValidIndirectBinary_t: {
      nn::Filter2D *f = (nn::Filter2D *)op_data->threads[t].filter2D;
      nn::MatMulBinary *aggr = (nn::MatMulBinary *)(f->aggregate_handler);
      aggr->setWeights(weights);
      nn::OT_binary *ot = (nn::OT_binary *)(f->ot_handler);
      ot->setThresholds(multipliers_and_biases);
    } break;
    case BNNConv2DValidDirectInt8_t: {
      nn::Filter2D *f = (nn::Filter2D *)op_data->threads[t].filter2D;
      nn::MatMulBinaryDirectFn *aggr =
          (nn::MatMulBinaryDirectFn *)(f->aggregate_handler);
      aggr->setWeights(weights);
      nn::OT_int8_clamped *ot = (nn::OT_int8_clamped *)(f->ot_handler);
      ot->setOffsetsMultipliersAndBiases(multipliers_and_biases);
    } break;
    case BNNConv2DValidIndirectInt8_t: {
      nn::Filter2D *f = (nn::Filter2D *)op_data->threads[t].filter2D;
      nn::MatMulBinary *aggr = (nn::MatMulBinary *)(f->aggregate_handler);
      aggr->setWeights(weights);
      nn::OT_int8_clamped *ot = (nn::OT_int8_clamped *)(f->ot_handler);
      ot->setOffsetsMultipliersAndBiases(multipliers_and_biases);
    } break;
    }

    thread_data[t].f = op_data->threads[t].filter2D;
    dispatcher->AddTask(reinterpret_cast<void *>(&thread_data[t]));
  }

  dispatcher->JoinTasks();
  // dispatcher->Invoke(dispatcher_args, n_threads);

  return kTfLiteOk;
}

} // namespace conv_v2

TfLiteRegistration *Register_Conv2D_V2() {
  static TfLiteRegistration r = {conv_v2::Init, nullptr, conv_v2::Prepare,
                                 conv_v2::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
