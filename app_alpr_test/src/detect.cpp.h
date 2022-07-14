// This file is generated. Do not edit.
// Generated on: 05.07.2022 14:53:15

#ifndef detect_GEN_H
#define detect_GEN_H

#include "tensorflow/lite/c/common.h"

// Sets up the model with init and prepare steps.
TfLiteStatus detect_init(void *flash_data = nullptr);
// Returns the input tensor with the given index.
TfLiteTensor *detect_input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *detect_output(int index);
// Runs inference for the model.
TfLiteStatus detect_invoke();

// Returns the number of input tensors.
inline size_t detect_inputs() {
  return 1;
}
// Returns the number of output tensors.
inline size_t detect_outputs() {
  return 2;
}

inline void *detect_input_ptr(int index) {
  return detect_input(index)->data.data;
}
inline size_t detect_input_size(int index) {
  return detect_input(index)->bytes;
}
inline int detect_input_dims_len(int index) {
  return detect_input(index)->dims->data[0];
}
inline int *detect_input_dims(int index) {
  return &detect_input(index)->dims->data[1];
}

inline void *detect_output_ptr(int index) {
  return detect_output(index)->data.data;
}
inline size_t detect_output_size(int index) {
  return detect_output(index)->bytes;
}
inline int detect_output_dims_len(int index) {
  return detect_output(index)->dims->data[0];
}
inline int *detect_output_dims(int index) {
  return &detect_output(index)->dims->data[1];
}

#endif
