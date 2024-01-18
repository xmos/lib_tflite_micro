#ifndef XCORE_COMMON_H_
#define XCORE_COMMON_H_

#include <cassert>
#include <cstdint>

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void calculateThreadSplit(int &tc, int split_size, int split_start[], int split_end[]);

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite

#endif // XCORE_COMMON_H_
