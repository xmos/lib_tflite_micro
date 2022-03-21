// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1
//#include "tensorflow/lite/micro/all_ops_resolver.h"

#include "inference_engine.h"
#include <cstdio>
//#include "tensorflow/lite/micro/kernels/xcore/xcore_ops.h"
//#include "tensorflow/lite/micro/recording_micro_allocator.

//*****************************************
// C-API callable from Python
//*****************************************

void add_lib_vision_ops(
    tflite::MicroMutableOpResolver<XTFLM_OPERATORS> *resolver) {
  resolver->AddAddN();
  resolver->AddCast();
  resolver->AddFloor();
  resolver->AddGreater();
  resolver->AddGreaterEqual();
  resolver->AddLess();
  resolver->AddLessEqual();
  resolver->AddLogicalAnd();
  resolver->AddMul();
  resolver->AddNeg();
  resolver->AddPadV2();
  resolver->AddReduceMax();
  resolver->AddResizeBilinear();
  resolver->AddResizeNearestNeighbor();
  resolver->AddRound();
  resolver->AddStridedSlice();
  resolver->AddSub();
  resolver->AddTranspose();
  resolver->AddGather();
  resolver->AddFullyConnected();
  resolver->AddTranspose();
  resolver->AddSlice();
  resolver->AddSplit();
  resolver->AddPack();
  resolver->AddUnpack();
  resolver->AddTanh();
  resolver->AddSplitV();
  resolver->AddShape();
}

extern "C" {
inference_engine *new_interpreter(size_t max_model_size) {
  inference_engine *ie =
      (inference_engine *)calloc(sizeof(inference_engine), 1);
  uint32_t *model_content = (uint32_t *)calloc(max_model_size, 1);

  struct tflite_micro_objects *s0 = new struct tflite_micro_objects;

  auto *resolver = inference_engine_initialize(ie, model_content,
                                               max_model_size, nullptr, 0, s0);

  resolver->AddDequantize();
  resolver->AddSoftmax();
  resolver->AddMean();
  resolver->AddPad();
  resolver->AddReshape();
  resolver->AddConcatenation();
  resolver->AddAdd();
  resolver->AddMinimum();
  resolver->AddMaximum();
  resolver->AddRelu();
  resolver->AddLogistic();
  resolver->AddConv2D();
  resolver->AddQuantize();
  resolver->AddDepthwiseConv2D();
  resolver->AddCustom(tflite::ops::micro::xcore::Conv2D_V2_OpCode,
                      tflite::ops::micro::xcore::Register_Conv2D_V2());
  resolver->AddCustom(tflite::ops::micro::xcore::Load_Flash_OpCode,
                      tflite::ops::micro::xcore::Register_LoadFromFlash());
  resolver->AddCustom(tflite::ops::micro::xcore::Bsign_8_OpCode,
                      tflite::ops::micro::xcore::Register_BSign_8());

  add_lib_vision_ops(resolver);

  return ie;
}

void delete_interpreter(inference_engine *ie) {
  inference_engine_unload_model(ie);
  free(ie->xtflm);
  free(ie);
}

int initialize(inference_engine *ie, const char *model_content,
               size_t model_content_size,
               const char *param_content) {
  // We need to keep a copy of the model content
  inference_engine_unload_model(ie);
  uint32_t *m = (uint32_t *)model_content;
  memcpy(ie->memory_primary, m, model_content_size);
  int r = inference_engine_load_model(ie, model_content_size, ie->memory_primary,
                                      (void *)param_content);
  return kTfLiteOk;
}

void print_memory_plan(inference_engine *ie) {
  ie->xtflm->interpreter->PrintMemoryPlan();
}

size_t inputs_size(inference_engine *ie) { return ie->inputs; }

size_t outputs_size(inference_engine *ie) { return ie->outputs; }

size_t get_input_tensor_size(inference_engine *ie, int tensor_index) {
  return ie->input_sizes[tensor_index];
}

size_t get_output_tensor_size(inference_engine *ie, int tensor_index) {
  return ie->output_sizes[tensor_index];
}

size_t arena_used_bytes(inference_engine *ie) {
  return ie->xtflm->interpreter->arena_used_bytes();
}

int set_input_tensor(inference_engine *ie, size_t tensor_index,
                     const void *value, const int size) {
  memcpy(ie->input_buffers[tensor_index], value, size);
  return 0;
}

int get_output_tensor(inference_engine *ie, size_t tensor_index, void *value,
                      const int size) {
  memcpy(value, ie->output_buffers[tensor_index], size);
  return 0;
}

int get_input_tensor(inference_engine *ie, size_t tensor_index, void *value,
                      const int size) {
  memcpy(value, ie->input_buffers[tensor_index], size);
  return 0;
}

int invoke(inference_engine *ie) { return interp_invoke_par_4(ie); }

size_t get_tensor_details_buffer_sizes(inference_engine *ie,
                                       size_t tensor_index, size_t *dims,
                                       size_t *scales, size_t *zero_points) {
  return ie->xtflm->interpreter->GetTensorDetailsBufferSizes(
      tensor_index, dims, scales, zero_points);
}

int get_tensor_details(inference_engine *ie, size_t tensor_index, char *name,
                       int name_len, int *shape, int *type, float *scale,
                       int *zero_point) {
  return ie->xtflm->interpreter->GetTensorDetails(
      tensor_index, name, name_len, shape, type, scale, zero_point);
}

size_t get_error(inference_engine *ie, char *msg) {
  std::strcpy(msg, (const char *)ie->debug_log_buffer);
  return strlen(msg);
}

size_t input_tensor_index(inference_engine *ie, size_t input_index) {
  return ie->xtflm->interpreter->input_tensor_index(input_index);
}

size_t output_tensor_index(inference_engine *ie, size_t output_index) {
  return ie->xtflm->interpreter->output_tensor_index(output_index);
}

} // extern "C"
