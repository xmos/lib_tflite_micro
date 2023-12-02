// XTLM_OPERATORS must be 200 as we use PythonOpsResolver in
// tflite micro compiler.
// PythonOpsResolver is defined as MicroMutableOpResolver<200> in
// https://github.com/tensorflow/tflite-micro/blob/main/python/tflite_micro/python_ops_resolver.h
#define XTFLM_OPERATORS (200)
#define NUM_OUTPUT_TENSORS (40)
#define NUM_INPUT_TENSORS (40)
#define MAX_DEBUG_LOG_LENGTH (1024)
#define AISRV_GPIO_LENGTH (4)
