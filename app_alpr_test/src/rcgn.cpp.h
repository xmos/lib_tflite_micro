// This file is generated. Do not edit.
// Generated on: 05.07.2022 14:47:15

#ifndef rcgn_GEN_H
#define rcgn_GEN_H

#include "tensorflow/lite/c/common.h"

// Sets up the model with init and prepare steps.
TfLiteStatus rcgn_init(void *flash_data = nullptr);
// Returns the input tensor with the given index.
TfLiteTensor *rcgn_input(int index);
// Returns the output tensor with the given index.
TfLiteTensor *rcgn_output(int index);
// Runs inference for the model.
TfLiteStatus rcgn_invoke();

// Returns the number of input tensors.
inline size_t rcgn_inputs() {
  return 1;
}
// Returns the number of output tensors.
inline size_t rcgn_outputs() {
  return 1;
}

inline void *rcgn_input_ptr(int index) {
  return rcgn_input(index)->data.data;
}
inline size_t rcgn_input_size(int index) {
  return rcgn_input(index)->bytes;
}
inline int rcgn_input_dims_len(int index) {
  return rcgn_input(index)->dims->data[0];
}
inline int *rcgn_input_dims(int index) {
  return &rcgn_input(index)->dims->data[1];
}

inline void *rcgn_output_ptr(int index) {
  return rcgn_output(index)->data.data;
}
inline size_t rcgn_output_size(int index) {
  return rcgn_output(index)->bytes;
}
inline int rcgn_output_dims_len(int index) {
  return rcgn_output(index)->dims->data[0];
}
inline int *rcgn_output_dims(int index) {
  return &rcgn_output(index)->dims->data[1];
}

#endif
