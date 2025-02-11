#include "tensorflow/lite/c/common.h"
#include "xcore_custom_options.h"
#include "xcore_utils.h"

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {
namespace transpose {

using tflite_micro::micro::GetEvalInput;
using tflite_micro::micro::GetEvalOutput;
using tflite_micro::micro::GetTensorData;

constexpr int kTransposeDims = 4; // Exactly 4 dimensions as specified

struct TransposeOpData {
  int32_t t_shape[kTransposeDims];
  int32_t offsets[kTransposeDims];
};

void *Init(TfLiteContext *context, const char *buffer, size_t length) {
  auto op_data = construct_persistent_object<TransposeOpData>(context);
  auto parser = CustomOptionParser(buffer, length);

  auto t_shape_vector = parser.parseNamedCustomOption("s").AsVector();
  auto offsets_vector = parser.parseNamedCustomOption("o").AsVector();

  for (int i = 0; i < kTransposeDims; ++i) {
    op_data->t_shape[i] = t_shape_vector[i].AsInt32();
    op_data->offsets[i] = offsets_vector[i].AsInt32();
  }

  return op_data;
}

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  // No preparation needed
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  auto *op_data = static_cast<TransposeOpData *>(node->user_data);

  const int32_t *t_shape = op_data->t_shape;
  const int32_t *offsets = op_data->offsets;

  const int8_t *input_data =
      GetTensorData<int8_t>(GetEvalInput(context, node, 0));
  int8_t *output_data = GetTensorData<int8_t>(GetEvalOutput(context, node, 0));

  // TODO: 1. Optimise by pre-computing increments
  // TODO: 2. Dereference t_shape in advance
  // TODO: 3. Multi-threading
  for (int i0 = 0; i0 < t_shape[0]; ++i0) {
    const int j0 = i0 * offsets[0];
    for (int i1 = 0; i1 < t_shape[1]; ++i1) {
      const int j1 = j0 + i1 * offsets[1];
      for (int i2 = 0; i2 < t_shape[2]; ++i2) {
        const int j2 = j1 + i2 * offsets[2];
        for (int i3 = 0; i3 < t_shape[3]; ++i3) {
          const int j3 = j2 + i3 * offsets[3];
          *output_data++ = input_data[j3];
        }
      }
    }
  }

  return kTfLiteOk;
}

} // namespace transpose

TFLMRegistration *Register_XC_transpose() {
  static TFLMRegistration r = {transpose::Init, nullptr, transpose::Prepare,
                               transpose::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
