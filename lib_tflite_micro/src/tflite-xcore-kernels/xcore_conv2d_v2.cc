// Copyright (c) 2023, XMOS Ltd, All rights reserved

#include "../thread_call.h"
#include "lib_nn/api/AbstractKernel.hpp"
#include "lib_nn/api/AggregateFn.hpp"
#include "lib_nn/api/MemCpyFn.hpp"
#include "lib_nn/api/OutputTransformFn.hpp"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "xcore_config.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"
extern "C" {
#include "lib_nn/api/nn_operator.h"
#include "lib_nn/api/expand_8_to_16.h"
#include "lib_nn/api/output_transform_fn_int16.h"
}

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace conv_v2 {

// -------------------------------------------------------------------- //
// thread data type and worker functions
// -------------------------------------------------------------------- //

struct Conv2DShared {
  int8_t *X;
  int8_t *Y;
  nn::conv_params_t *conv_params;
  int8_t *weights;
  int16_t *muls_and_biases;
  bool isDepthwise;
};

extern "C" {
// TODO
#pragma stackfunction 1000
void conv2d_v2_thread_worker(void *shard, void *scrtch, void *kp) {
  nn::abstract_kernel_params_t *akparams = (nn::abstract_kernel_params_t *)kp;
  auto scratch = static_cast<int8_t *>(scrtch);
  auto shared = static_cast<Conv2DShared *>(shard);
  execute(shared->Y, shared->X + akparams->input_offset, shared->conv_params,
          akparams, shared->weights, shared->muls_and_biases,
          /*isConv=*/!shared->isDepthwise, scratch);
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
  BNNConv2DValidIndirectInt8_t,
  Conv2DValidDirectI16_t,
  // DepthwiseConv2DValidDirectI16_t
};

enum OT_Type { Group, Channelwise };

/**
 * @brief This describes the memory requirements of a worker thread. It also
 * includes an array of the work to be done by said worker.
 *
 */
struct Conv2DThreadInfo {
  nn::abstract_kernel_params_t *kparams;
};

// This is the struct that contains the data required to fully describe the work
// that the operator will perform. It needs to descibe the work for T threads.
// That means it must contain:
// - T sets of work, i.e. a list of jobs for each thread.
// - T scratch allocations, i.e. an amount of scratch memory for each thread.
struct Conv2DOpData
    : XCoreOpData { // Inherits the operator name field from XCoreOpData
  size_t thread_count;
  size_t scratch_size;
#ifdef TFLMC_CONV2D_PROFILE
  int evalStartTime;
  int threadsStartTime;
  int threadsDoneTime;
#endif
  Conv2DThreadInfo *threads;
  nn::conv_params_t conv_params; // The job to be done by this thread
  KernelType kt;
  int i16_expanded_weights_scratch_index;
};

// -------------------------------------------------------------------- //
// op function implementations
// -------------------------------------------------------------------- //

template <typename T> T *getDeserializedParams(const uint8_t *data) {
  // The flexbuffer data passed along from the compiler for xc_conv2d is
  // carefully aligned to four bytes so that we can directly access it.
  // This is done in TranslateToCustomOp.cpp in the compiler.
  assert(((uintptr_t)data & 0x3) == 0);
  T *param = (T *)data;
  return param;
}

// Construct Filter2D threads
template <typename MfStructType>
void ConstructFilter2DsImpl(Conv2DOpData *op_data, TfLiteContext *context,
                            const uint8_t *memcpy_fn_data,
                            const uint8_t *agg_fn_data,
                            flexbuffers::Vector &ak_params_vec) {
  if (std::is_same<MfStructType, nn::memcpyfn_deref_params_t>::value) {
    op_data->conv_params.memcopy_fn = (nn::MemFnType)nn::memcpyfn_deref;
  } else if (std::is_same<MfStructType,
                          nn::memcpyfn_imtocol_valid_params_t>::value) {
    op_data->conv_params.memcopy_fn = (nn::MemFnType)nn::memcpyfn_imtocol_valid;
  } else if (std::is_same<MfStructType,
                          nn::memcpyfn_imtocol_padded_params_t>::value) {
    op_data->conv_params.memcopy_fn =
        (nn::MemFnType)nn::memcpyfn_imtocol_padded;
  } else {
    assert(false);
  }

  op_data->conv_params.mem_p =
      getDeserializedParams<MfStructType>(memcpy_fn_data);

  // For each thread, we have a different set of abstract kernel params which we
  // extract here
  // We reuse the other params
  for (int t = 0; t < op_data->thread_count; ++t) {
    op_data->threads[t].kparams =
        getDeserializedParams<nn::abstract_kernel_params_t>(
            ak_params_vec[t].AsBlob().data());
  }
}

// Specialised for the binary output cases
// Forwards into the ConstructFilter2DsImpl implementation function
template <typename MfStructType, typename AggStructType, typename OtStructType,
          bool binaryOutput>
void ConstructFilter2Ds(Conv2DOpData *op_data, TfLiteContext *context,
                        const uint8_t *memcpy_fn_data,
                        const uint8_t *agg_fn_data, const uint8_t *ot_fn_data,
                        flexbuffers::Vector &ak_params_vec) {
  if (std::is_same<OtStructType, nn::otfn_int8_clamped_params_t>::value) {
    op_data->conv_params.output_transform_fn =
        (nn::OtFnType)nn::otfn_int8_clamped;
    op_data->conv_params.ot_p = getDeserializedParams<OtStructType>(ot_fn_data);
  } else if (std::is_same<OtStructType, std::nullptr_t>::value) {
    op_data->conv_params.output_transform_fn = (nn::OtFnType)nn::otfn_binary;
  } else {
    assert(false);
  }

  if (std::is_same<AggStructType, nn::mat_mul_direct_params_t>::value) {
    op_data->conv_params.aggregate_fn =
        (nn::AggFnType)nn::mat_mul_direct_binary;
  } else if (std::is_same<AggStructType, nn::mat_mul_generic_params_t>::value) {
    op_data->conv_params.aggregate_fn =
        (nn::AggFnType)nn::mat_mul_generic_binary;
  } else {
    assert(false);
  }

  op_data->conv_params.agg_p =
      getDeserializedParams<AggStructType>(agg_fn_data);

  ConstructFilter2DsImpl<MfStructType>(op_data, context,
                                       memcpy_fn_data, agg_fn_data,
                                       ak_params_vec);
}

// Forwards into the ConstructFilter2DsImpl implementation function
// For binary output, we specialise that case separately
template <typename MfStructType, typename AggStructType, typename OtStructType>
void ConstructFilter2Ds(Conv2DOpData *op_data, TfLiteContext *context,
                        const uint8_t *memcpy_fn_data,
                        const uint8_t *agg_fn_data, const uint8_t *ot_fn_data,
                        flexbuffers::Vector &ak_params_vec,
                        bool isI16Conv = false) {
  if (std::is_same<OtStructType, nn::otfn_int8_channelwise_params_t>::value) {
    op_data->conv_params.output_transform_fn =
        (nn::OtFnType)nn::otfn_int8_channelwise;
  } else if (std::is_same<OtStructType, nn::otfn_int8_params_t>::value) {
    op_data->conv_params.output_transform_fn = (nn::OtFnType)nn::otfn_int8;
  } else if (std::is_same<OtStructType, otfn_int16_params_t>::value) {
    op_data->conv_params.output_transform_fn =
        (nn::OtFnType)output_transform_fn_int16;
  } else {
    assert(false);
  }

  if (isI16Conv &&
      std::is_same<AggStructType, nn::mat_mul_direct_params_t>::value) {
    op_data->conv_params.aggregate_fn = (nn::AggFnType)nn::mat_mul_direct_int16;
  } else if (std::is_same<AggStructType, nn::mat_mul_direct_params_t>::value) {
    op_data->conv_params.aggregate_fn = (nn::AggFnType)nn::mat_mul_direct_int8;
  } else if (std::is_same<AggStructType, nn::mat_mul_generic_params_t>::value) {
    op_data->conv_params.aggregate_fn = (nn::AggFnType)nn::mat_mul_generic_int8;
  } else if (std::is_same<AggStructType,
                          nn::mat_mul_dw_direct_params_t>::value) {
    op_data->conv_params.aggregate_fn = (nn::AggFnType)nn::mat_mul_dw_direct;
  } else {
    assert(false);
  }

  op_data->conv_params.ot_p = getDeserializedParams<OtStructType>(ot_fn_data);
  op_data->conv_params.agg_p =
      getDeserializedParams<AggStructType>(agg_fn_data);

  ConstructFilter2DsImpl<MfStructType>(op_data, context,
                                       memcpy_fn_data, agg_fn_data,
                                       ak_params_vec);
}

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  TFLITE_DCHECK(buffer != nullptr);
  auto op_data = construct_persistent_object<Conv2DOpData>(context);
  auto parser = CustomOptionParser(buffer, length);

  KernelType kt = (KernelType)parser.parseNamedCustomOption("k").AsInt32();
  op_data->kt = kt;
  const uint8_t *memcpy_fn_data =
      parser.parseNamedCustomOption("mp").AsBlob().data();
  const uint8_t *agg_fn_data =
      parser.parseNamedCustomOption("a").AsBlob().data();
  const uint8_t *ot_fn_data =
      parser.parseNamedCustomOption("o").AsBlob().data();
  op_data->scratch_size = parser.parseNamedCustomOption("s").AsInt32();
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

  op_data->threads =
      static_cast<Conv2DThreadInfo *>(context->AllocatePersistentBuffer(
          context, op_data->thread_count * sizeof(Conv2DThreadInfo)));

  switch (kt) {
  case Conv2DValidDirect_t: {
    if (ot_type == Channelwise) {
      ConstructFilter2Ds<nn::memcpyfn_deref_params_t,
                         nn::mat_mul_direct_params_t,
                         nn::otfn_int8_channelwise_params_t>(
          op_data, context, memcpy_fn_data, agg_fn_data,
          ot_fn_data, ak_params_vec);
    } else {
      ConstructFilter2Ds<nn::memcpyfn_deref_params_t,
                         nn::mat_mul_direct_params_t, nn::otfn_int8_params_t>(
          op_data, context, memcpy_fn_data, agg_fn_data,
          ot_fn_data, ak_params_vec);
    }
    op_data->name = "XC_Conv2DValidDir";
  } break;
  case Conv2DValidIndirect_t: {
    if (ot_type == Channelwise) {
      ConstructFilter2Ds<nn::memcpyfn_imtocol_valid_params_t,
                         nn::mat_mul_generic_params_t,
                         nn::otfn_int8_channelwise_params_t>(
          op_data, context, memcpy_fn_data, agg_fn_data,
          ot_fn_data, ak_params_vec);
    } else {
      ConstructFilter2Ds<nn::memcpyfn_imtocol_valid_params_t,
                         nn::mat_mul_generic_params_t, nn::otfn_int8_params_t>(
          op_data, context, memcpy_fn_data, agg_fn_data,
          ot_fn_data, ak_params_vec);
    }
    op_data->name = "XC_Conv2DValidInd";
  } break;
  case Conv2DPaddedIndirect_t: {
    if (ot_type == Channelwise) {
      ConstructFilter2Ds<nn::memcpyfn_imtocol_padded_params_t,
                         nn::mat_mul_generic_params_t,
                         nn::otfn_int8_channelwise_params_t>(
          op_data, context, memcpy_fn_data, agg_fn_data,
          ot_fn_data, ak_params_vec);
    } else {
      ConstructFilter2Ds<nn::memcpyfn_imtocol_padded_params_t,
                         nn::mat_mul_generic_params_t, nn::otfn_int8_params_t>(
          op_data, context, memcpy_fn_data, agg_fn_data,
          ot_fn_data, ak_params_vec);
    }
    op_data->name = "XC_Conv2DPadInd";
  } break;
  case DepthwiseConv2DValidDirect_t: {
    if (ot_type == Channelwise) {
      ConstructFilter2Ds<nn::memcpyfn_deref_params_t,
                         nn::mat_mul_dw_direct_params_t,
                         nn::otfn_int8_channelwise_params_t>(
          op_data, context, memcpy_fn_data, agg_fn_data,
          ot_fn_data, ak_params_vec);
    } else {
      ConstructFilter2Ds<nn::memcpyfn_deref_params_t,
                         nn::mat_mul_dw_direct_params_t,
                         nn::otfn_int8_params_t>(op_data, context,
                                                 memcpy_fn_data, agg_fn_data,
                                                 ot_fn_data, ak_params_vec);
    }
    op_data->name = "XC_DWConv2DValidInd";
  } break;
  case DepthwiseConv2DPaddedIndirect_t: {
    if (ot_type == Channelwise) {
      ConstructFilter2Ds<nn::memcpyfn_imtocol_padded_params_t,
                         nn::mat_mul_dw_direct_params_t,
                         nn::otfn_int8_channelwise_params_t>(
          op_data, context, memcpy_fn_data, agg_fn_data,
          ot_fn_data, ak_params_vec);
    } else {
      ConstructFilter2Ds<nn::memcpyfn_imtocol_padded_params_t,
                         nn::mat_mul_dw_direct_params_t,
                         nn::otfn_int8_params_t>(op_data, context,
                                                 memcpy_fn_data, agg_fn_data,
                                                 ot_fn_data, ak_params_vec);
    }
    op_data->name = "XC_DWConv2DPadInd";
  } break;
  case BNNConv2DValidDirectBinary_t: {
    ConstructFilter2Ds<nn::memcpyfn_deref_params_t, nn::mat_mul_direct_params_t,
                       std::nullptr_t, /*binaryOutput=*/true>(
        op_data, context, memcpy_fn_data, agg_fn_data, ot_fn_data,
        ak_params_vec);
    op_data->name = "XC_BNNValidDirBin";
  } break;
  case BNNConv2DValidIndirectBinary_t: {
    ConstructFilter2Ds<nn::memcpyfn_imtocol_valid_params_t,
                       nn::mat_mul_generic_params_t, std::nullptr_t,
                       /*binaryOutput=*/true>(op_data, context,
                                              memcpy_fn_data, agg_fn_data,
                                              ot_fn_data, ak_params_vec);
    op_data->name = "XC_BNNValidIndBin";
  } break;
  case BNNConv2DValidDirectInt8_t: {
    ConstructFilter2Ds<nn::memcpyfn_deref_params_t, nn::mat_mul_direct_params_t,
                       nn::otfn_int8_clamped_params_t, /*binaryOutput=*/true>(
        op_data, context, memcpy_fn_data, agg_fn_data, ot_fn_data,
        ak_params_vec);
    op_data->name = "XC_BNNValidDirInt8";
  } break;
  case BNNConv2DValidIndirectInt8_t: {
    ConstructFilter2Ds<nn::memcpyfn_imtocol_valid_params_t,
                       nn::mat_mul_generic_params_t,
                       nn::otfn_int8_clamped_params_t, /*binaryOutput=*/true>(
        op_data, context, memcpy_fn_data, agg_fn_data, ot_fn_data,
        ak_params_vec);
    op_data->name = "XC_BNNValidIndInt8";
  } break;
  case Conv2DValidDirectI16_t: {
    ConstructFilter2Ds<nn::memcpyfn_deref_params_t, nn::mat_mul_direct_params_t,
                       otfn_int16_params_t>(
        op_data, context, memcpy_fn_data, agg_fn_data, ot_fn_data,
        ak_params_vec, /*isI16Conv=*/true);
    op_data->name = "XC_Conv2DValidDirI16";
  } break;
  }
  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteEvalTensor *input_tensor =
      tflite_micro::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output_tensor =
      tflite_micro::micro::GetEvalOutput(context, node, 0);
  const TfLiteEvalTensor *weights_tensor =
      tflite_micro::micro::GetEvalInput(context, node, 1);
  const TfLiteEvalTensor *multipliers_and_biases_tensor =
      tflite_micro::micro::GetEvalInput(context, node, 2);

  int8_t *output =
      (int8_t *)tflite_micro::micro::GetTensorData<int8_t>(output_tensor);
  const TfLiteEvalTensor *partial_output_tensor =
      tflite_micro::micro::GetEvalInput(context, node, 3);
  // Copy the partial output into the output tensor only if it is not NULL
  if (partial_output_tensor) {
    int8_t *partial_output =
        (int8_t *)tflite_micro::micro::GetTensorData<int8_t>(partial_output_tensor);
    size_t sizeof_tensor_type;
    TfLiteTypeSizeOf(partial_output_tensor->type, &sizeof_tensor_type);
    int size = tflite_micro::micro::GetTensorShape(partial_output_tensor).FlatSize();
    memcpy((int8_t *)output, (int8_t *)partial_output,
           size * sizeof_tensor_type);
  }

  const TfLiteEvalTensor *scratch_buffer_tensor =
      tflite_micro::micro::GetEvalInput(context, node, 4);
  int8_t *scratch_buffer = nullptr;
  if (scratch_buffer_tensor) {
    scratch_buffer = const_cast<int8_t *>(tflite_micro::micro::GetTensorData<int8_t>(scratch_buffer_tensor));
  }

  MicroContext *micro_context = GetMicroContext(context);
  xc_context_config_t *xc_config = reinterpret_cast<xc_context_config_t *>(
      micro_context->external_context());

  auto *op_data = reinterpret_cast<Conv2DOpData *>(node->user_data);
  int n_threads = op_data->thread_count;

#ifdef TFLMC_CONV2D_PROFILE
  asm volatile("gettime %0" : "=r"(op_data->evalStartTime));
#endif

  int8_t *weights =
      (int8_t *)tflite_micro::micro::GetTensorData<int8_t>(weights_tensor);
  int16_t *multipliers_and_biases =
      (int16_t *)tflite_micro::micro::GetTensorData<int16_t>(
          multipliers_and_biases_tensor);

  int8_t *thread_scratch[XCORE_MAX_NUM_THREADS];
  Conv2DShared shared_data;
  shared_data.X = (int8_t *)tflite_micro::micro::GetTensorData<int8_t>(input_tensor);
  shared_data.Y = output;
  shared_data.conv_params = &op_data->conv_params;
  if (op_data->kt == DepthwiseConv2DValidDirect_t ||
      op_data->kt == DepthwiseConv2DPaddedIndirect_t) {
    shared_data.isDepthwise = true;
  } else {
    shared_data.isDepthwise = false;
  }

  // expand weights for int16
  if (op_data->kt == Conv2DValidDirectI16_t) {
    // scratch buffer contains scratch space for each thread and for expanding i16 weights
    int16_t *i16_expanded_weights_scratch = (int16_t *)&scratch_buffer[n_threads * op_data->scratch_size];
    expand_8_to_16(i16_expanded_weights_scratch, weights,
                   tflite_micro::micro::GetTensorShape(weights_tensor).FlatSize());
    shared_data.weights = (int8_t *)i16_expanded_weights_scratch;
  } else {
    shared_data.weights = weights;
  }
  shared_data.muls_and_biases = multipliers_and_biases;
  if (op_data->scratch_size) {
    for (int t = 0; t < n_threads; ++t) {
        thread_scratch[t] = (int8_t *)&scratch_buffer[t * op_data->scratch_size];
    }
  }

  // todo - this second for-loop is unpleasant
  for (int t = 0; t < n_threads - 1; ++t) {
    thread_variable_setup(thread_scratch[t], op_data->threads[t].kparams,
                          xc_config->thread_info.thread_ids.id[t]);
  }

#ifdef TFLMC_CONV2D_PROFILE
  asm volatile("gettime %0" : "=r"(op_data->threadsStartTime));
#endif

  // Now set up shared data, shared function pointer, and data for final thread.
  thread_call((void *)&shared_data, thread_scratch[n_threads - 1],
              op_data->threads[n_threads - 1].kparams,
              (thread_function_pointer_t)conv2d_v2_thread_worker,
              &xc_config->thread_info);

#ifdef TFLMC_CONV2D_PROFILE
  asm volatile("gettime %0" : "=r"(op_data->threadsDoneTime));
#endif

  return kTfLiteOk;
}

} // namespace conv_v2

TFLMRegistration *Register_XC_conv2d_v2() {
  static TFLMRegistration r = {conv_v2::Init, nullptr, conv_v2::Prepare,
                               conv_v2::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
