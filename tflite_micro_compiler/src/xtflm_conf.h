// XTLM_OPERATORS must be 128 as we use AllOpsResolver in
// tflite micro compiler.
// AllOpsResolver is defined as MicroMutableOpResolver<128> in
// https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/all_ops_resolver.h
#define XTFLM_OPERATORS (128)
#define NUM_OUTPUT_TENSORS (40)
#define NUM_INPUT_TENSORS (40)
#define MAX_DEBUG_LOG_LENGTH (1024)
#define AISRV_GPIO_LENGTH (4)
