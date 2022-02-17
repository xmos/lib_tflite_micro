// Copyright (c) 2021, XMOS Ltd, All rights reserved
#include "server_memory.h"
#include <cstdint>
#include <cstring>

#if !defined(XTFLM_DISABLED)

#define TENSOR_ARENA_BYTES_0 (20224000)

// Because of bug in xgdb we make this array tiny, knowing we own external
// memory otherwise xgdb spends hours loading this array
// TODO - fix when bug is fixed.
//__attribute__((section(".ExtMem_data")))
uint32_t *data_ext = (uint32_t *)0x10000000; //[TENSOR_ARENA_BYTES_0/sizeof(int)];
                                             //// engine 0, tile 1
#endif

void inference_engine_initialize_with_memory(inference_engine_t *ie) {
#if !defined(XTFLM_DISABLED)
  static struct tflite_micro_objects s0;
  memset(data_ext, 0, TENSOR_ARENA_BYTES_0);
  auto *resolver = inference_engine_initialize(
      ie, data_ext, TENSOR_ARENA_BYTES_0, nullptr, 0, &s0);

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
#endif
}
