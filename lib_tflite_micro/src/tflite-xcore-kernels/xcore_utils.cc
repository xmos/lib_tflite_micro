#include "xcore_utils.h"

namespace tflite_micro {
namespace ops {
namespace micro {
namespace xcore {

TfLiteStatus No_Op_Eval(TfLiteContext *context, TfLiteNode *node) {
  // Get Input/Output Tensors
  // const TfLiteEvalTensor *input = tflite_micro::micro::GetEvalInput(context, node, 0);
  // TfLiteEvalTensor *output = tflite_micro::micro::GetEvalOutput(context, node, 0);
  // // Pointers to data in In/Out Tensors
  // const int8_t *in_data = tflite_micro::micro::GetTensorData<int8_t>(input);
  // int8_t *out_data = tflite_micro::micro::GetTensorData<int8_t>(output);

  // size_t sizeof_tensor_type;
  // TfLiteTypeSizeOf(output->type, &sizeof_tensor_type);
  // int size = tflite_micro::micro::GetTensorShape(output).FlatSize();
  // memcpy((int8_t *)out_data, (int8_t *)in_data, size * sizeof_tensor_type);
  return kTfLiteOk;
} 

TFLMRegistration *Register_XC_no_op() {
  static TFLMRegistration r = {nullptr, nullptr, nullptr,
                                    No_Op_Eval};
  return &r;
}

size_t FetchBuffer(int8_t **dest, int8_t const *src, size_t size) {
  if (is_ram_address((uintptr_t)src)) {
    *dest = (int8_t *)src;
    return 0;
  } else {
    memload((void *)*dest, (void *)src, size);
    return size;
  }
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite_micro
