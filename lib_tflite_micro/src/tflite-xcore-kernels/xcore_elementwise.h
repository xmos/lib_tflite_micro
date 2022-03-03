#ifndef XCORE_ELEMENTWISE_H_
#define XCORE_ELEMENTWISE_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "xcore_custom_options.h"

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

template <typename TArgs> struct ElementwiseThreadData {
  int32_t start;
  int32_t element_count;
  const TArgs *args;
};

template <typename TMultiThreadedOpData>
void *ElementwiseInit(TfLiteContext *context, const char *buffer,
                      size_t length) {
  auto job_sizes =
      CustomOptionParser(buffer, length).parseElementwiseJobSizes();
  auto *op_data = construct_persistent_object<TMultiThreadedOpData>(context);

  // in this op we have one job per thread
  auto n_threads = job_sizes.size();
  op_data->threads.allocate(context, n_threads);
  int start_idx = 0;
  for (int j{0}; j < n_threads; j++) {
    auto job_size = job_sizes[j].AsInt32();
    op_data->threads.append({start_idx, job_size, &op_data->args});
    start_idx = start_idx + job_size;
  }

  return op_data;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite

#endif // XCORE_ELEMENTWISE_H_
