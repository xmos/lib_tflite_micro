// Copyright (c) 2019, XMOS Ltd, All rights reserved
#include "xcore_profiler.h"

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_time.h"

namespace tflite {
namespace micro {
namespace xcore {

void XCoreProfiler::Init(tflite::MicroAllocator *allocator,
                         size_t max_event_count) {
  max_event_count_ = max_event_count;
  event_durations_ = static_cast<uint32_t *>(
      allocator->AllocatePersistentBuffer(max_event_count * sizeof(uint32_t)));
}

uint32_t const *XCoreProfiler::GetEventDurations() { return event_durations_; }

size_t XCoreProfiler::GetNumEvents() { return event_count_; }

void XCoreProfiler::ClearEvents() { event_count_ = 0; }

uint32_t XCoreProfiler::BeginEvent(const char *tag) {
  TFLITE_DCHECK(tag);
  event_tag_ = tag;
  event_start_time_ = tflite::GetCurrentTimeTicks();
  return 0;
}

void XCoreProfiler::EndEvent(uint32_t event_handle) {
  int32_t event_end_time = tflite::GetCurrentTimeTicks();
  event_count_ = event_count_ % max_event_count_;
  // wrap if there are too many events
  event_durations_[event_count_++] = event_end_time - event_start_time_;
}

} // namespace xcore
} // namespace micro
} // namespace tflite
