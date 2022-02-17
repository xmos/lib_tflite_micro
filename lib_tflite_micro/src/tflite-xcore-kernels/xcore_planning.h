// Copyright (c) 2020, XMOS Ltd, All rights reserved
#ifndef XCORE_PLANNING_H_
#define XCORE_PLANNING_H_
#include <cassert>
#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "xcore_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

constexpr size_t kChannelGroupLength = (16);
constexpr size_t kBSOChannelGroupLength = (7 * kChannelGroupLength);
constexpr size_t kBSOChannelGroupBytes = (kBSOChannelGroupLength * 2);

typedef struct RowColRegion {
  int32_t top;
  int32_t left;
  int32_t rows;
  int32_t cols;
} RowColRegion;

typedef struct ChannelGroup {
  int32_t index;
  int32_t start;
  int32_t size;
} ChannelGroup;

class ExecutionPlan {
public:
  ExecutionPlan();
  ~ExecutionPlan() {}

  void SetNumThreads(int32_t n_threads) { n_threads_ = n_threads; }
  size_t GetNumThreads() { return n_threads_; }

  void SetWeightsScratchSize(size_t size);
  size_t GetWeightsScratchSize();

  void SetBiasScratchSize(size_t size);
  size_t GetBiasScratchSize();

  PersistentArray<RowColRegion> regions;
  PersistentArray<ChannelGroup> changrps;

private:
  size_t n_threads_;
  size_t bias_scratch_offset_;
  size_t bias_scratch_size_;
};

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite

#endif // XCORE_PLANNING_H_
