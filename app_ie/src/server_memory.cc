// Copyright (c) 2021, XMOS Ltd, All rights reserved
#include "server_memory.h"
#include <cstdint>
#include <cstring>

#if !defined(XTFLM_DISABLED)

#define TENSOR_ARENA_BYTES_0 (300000)

// Because of bug in xgdb we make this array tiny, knowing we own external
// memory otherwise xgdb spends hours loading this array
// TODO - fix when bug is fixed.
//__attribute__((section(".ExtMem_data")))
uint64_t data_ext [TENSOR_ARENA_BYTES_0/sizeof(uint64_t)];
                                             //// engine 0, tile 1
#endif

void inference_engine_initialize_with_memory(inference_engine_t *ie) {
#if !defined(XTFLM_DISABLED)
  static struct tflite_micro_objects s0;
  memset(data_ext, 0, TENSOR_ARENA_BYTES_0);
  auto *resolver = inference_engine_initialize(
      ie, (uint32_t *)data_ext, TENSOR_ARENA_BYTES_0, nullptr, 0, &s0);

  resolver->AddSoftmax();
  resolver->AddPad();
  resolver->AddMean();
  resolver->AddReshape();
  resolver->AddConcatenation();
  resolver->AddFullyConnected();
  resolver->AddAdd();
  resolver->AddMaxPool2D();
  resolver->AddAveragePool2D();
  resolver->AddPad();
  resolver->AddLogistic();
  resolver->AddConv2D();
  resolver->AddQuantize();
  resolver->AddDepthwiseConv2D();
  resolver->AddDequantize();
  resolver->AddCustom(tflite::ops::micro::xcore::Conv2D_V2_OpCode,
                      tflite::ops::micro::xcore::Register_Conv2D_V2());
  resolver->AddCustom(tflite::ops::micro::xcore::Strided_Slice_OpCode,
                      tflite::ops::micro::xcore::Register_Strided_Slice());
#endif
}
