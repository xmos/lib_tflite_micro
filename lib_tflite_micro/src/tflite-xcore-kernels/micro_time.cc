// Copyright (c) 2022, XMOS Ltd, All rights reserved

#include "tensorflow/lite/micro/micro_time.h"

extern "C" {
// These are headers from XMOS toolchain.
#include <platform.h>
#ifdef _TIME_H_
#define _clock_defined
#endif
#include <xcore/hwtimer.h>
}

namespace tflite {

int32_t ticks_per_second() { return PLATFORM_REFERENCE_HZ; }

int32_t GetCurrentTimeTicks() { return get_reference_time(); }

}  // namespace tflite
