#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

extern "C" {
#include "lib_nn/api/nn_operator.h"
}

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {
namespace argmax {

TfLiteStatus Prepare(TfLiteContext *context, TfLiteNode *node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext *context, TfLiteNode *node) {
  const TfLiteEvalTensor *input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor *output = tflite::micro::GetEvalOutput(context, node, 0);

  const RuntimeShape input_shape = tflite::micro::GetTensorShape(input);

  argmax_16(tflite::micro::GetTensorData<int32_t>(output),
            tflite::micro::GetTensorData<int16_t>(input),
            input_shape.FlatSize());

  return kTfLiteOk;
}

} // namespace argmax

TfLiteRegistration *Register_ArgMax_16() {
  static TfLiteRegistration r = {nullptr, nullptr, argmax::Prepare,
                                 argmax::Eval};
  return &r;
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite
