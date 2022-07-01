#ifndef TFLMCOMPILER_RECORDALLOCATIONS_H
#define TFLMCOMPILER_RECORDALLOCATIONS_H

#include <cinttypes>

#include "tensorflow/lite/schema/schema_generated.h"

namespace tflmc {

struct Allocation {
  ptrdiff_t offset;
  size_t len;
  int nodeIndex;
};

std::vector<Allocation> RecordAllocations(const tflite::Model *model,
                                          ptrdiff_t arena_size,
                                          size_t &max_scratch_buffer_size);

TfLiteEvalTensor *GetEvalTensor(tflite::MicroInterpreter *interpreter, int i);
TfLiteTensor *GetTensor(tflite::MicroInterpreter *interpreter, int i);

}  // namespace tflmc

#endif
