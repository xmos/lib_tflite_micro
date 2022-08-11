// Copyright (c) 2020, XMOS Ltd, All rights reserved

#include "xcore_interpreter.h"
#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"
#include "tensorflow/lite/micro/memory_planner/micro_memory_planner.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "xcore_utils.h"

namespace tflite {
namespace micro {
namespace xcore {

XCoreInterpreter::XCoreInterpreter(const tflite::Model *model,
                                   const tflite::MicroOpResolver &resolver,
                                   tflite::MicroAllocator *allocator,
                                   tflite::ErrorReporter *reporter,
                                   bool use_current_thread,
                                   XCoreProfiler *profiler)
    : tflite::MicroInterpreter(model, resolver, allocator, reporter, nullptr,
                               profiler) {
  this->model__ = model;
  this->error_reporter__ = reporter;
  this->allocator_ = allocator;
  if (profiler) {
    profiler->Init(allocator, model->subgraphs()->Get(0)->operators()->size());
  }
}

XCoreInterpreter *XCoreInterpreter::Create(
    uint8_t interpreter_buffer[], const tflite::Model *model,
    const tflite::MicroOpResolver &resolver, uint8_t *arena, size_t arena_size,
    tflite::ErrorReporter *reporter, bool use_current_thread,
    XCoreProfiler *profiler) {
  MicroAllocator *memory_allocator =
      MicroAllocator::Create(arena, arena_size, reporter);
  return new (interpreter_buffer)
      XCoreInterpreter(model, resolver, memory_allocator, reporter,
                       use_current_thread, profiler);
}

void XCoreInterpreter::PrintMemoryPlan() {
  allocator_->memory_planner()->PrintMemoryPlan();
}

TfLiteTensor *XCoreInterpreter::tensor(size_t tensor_index) {
  auto ctx = context();
  return ctx.GetTensor(&ctx, tensor_index);
}

// This function retrieves a node's name
// It is slightly awkward as it needs to retrieve the graph - which is private
// to the micro-interpreter.
const char *XCoreInterpreter::node_name(int sub_idx, int i) {
  auto ctx = context();
  TfLiteIntArray *arg = NULL;
  ctx.GetExecutionPlan(&ctx, &arg);
  MicroGraph *graph = (MicroGraph *)arg;
  void *user_data =
      graph->GetAllocations()[sub_idx].node_and_registrations[i].node.user_data;

  if (user_data != NULL) {
    struct tflite::ops::micro::XCoreOpData *x =
        (struct tflite::ops::micro::XCoreOpData *)user_data;
    return x->name;
  }
  return NULL;
}

TfLiteStatus XCoreInterpreter::GetTensorDetails(size_t tensor_index, char *name,
                                                int name_len, int *shape,
                                                int *type, float *scale,
                                                int32_t *zero_point) {
  const SubGraph *subgraph = model__->subgraphs()->Get(0);
  const Tensor *tensor_p = subgraph->tensors()->Get(tensor_index);

  if (tensor_p == nullptr) {
    return kTfLiteError;
  }

  if (tensor_p->name()) {
    std::strncpy(name, tensor_p->name()->c_str(), name_len);
  }

  auto *shape_vector = tensor_p->shape();
  if (shape_vector) {
    for (int i = 0; i < shape_vector->Length(); i++) {
      shape[i] = shape_vector->Get(i);
    }
  }

  scale[0] = 0.0;
  zero_point[0] = 0;

  ConvertTensorType(tensor_p->type(), (TfLiteType *)type, error_reporter__);
  const tflite::QuantizationParameters *quantization_params =
      tensor_p->quantization();
  if (quantization_params) {
    auto *scale_vector = quantization_params->scale();
    if (scale_vector) {
      for (int i = 0; i < scale_vector->Length(); i++) {
        scale[i] = scale_vector->Get(i);
      }
    }

    auto *zero_points_vector = quantization_params->zero_point();
    if (zero_points_vector) {
      for (int i = 0; i < zero_points_vector->Length(); i++) {
        zero_point[i] = zero_points_vector->Get(i);
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus XCoreInterpreter::GetTensorDetailsBufferSizes(
    size_t tensor_index, size_t *dims, size_t *scales, size_t *zero_points) {
  const SubGraph *subgraph = model__->subgraphs()->Get(0);
  const Tensor *tensor_p = subgraph->tensors()->Get(tensor_index);

  if (tensor_p == nullptr) {
    return kTfLiteError;
  }

  *dims = 0;
  auto *shape_vector = tensor_p->shape();
  if (shape_vector) {
    *dims = shape_vector->Length();
  }

  *scales = 1;
  *zero_points = 1;
  const tflite::QuantizationParameters *quantization_params =
      tensor_p->quantization();
  if (quantization_params) {
    auto *scale_vector = quantization_params->scale();
    if (scale_vector) {
      *scales = scale_vector->Length();
    }

    auto *zero_points_vector = quantization_params->zero_point();
    if (zero_points_vector) {
      *zero_points = zero_points_vector->Length();
    }
  }
  return kTfLiteOk;
}

size_t XCoreInterpreter::input_tensor_index(size_t input_index) {
  const SubGraph *subgraph = model__->subgraphs()->Get(0);
  return subgraph->inputs()->Get(input_index);
}

size_t XCoreInterpreter::output_tensor_index(size_t output_index) {
  const SubGraph *subgraph = model__->subgraphs()->Get(0);
  return subgraph->outputs()->Get(output_index);
}

} // namespace xcore
} // namespace micro
} // namespace tflite
