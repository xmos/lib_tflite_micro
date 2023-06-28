#include "stdio.h"

#include "xcore_optimised_mobilenetV2_input_160x160x3_alpha_1.tflite.h"

int main(void) {
  model_init(NULL);
  int8_t *inputs = (int8_t *)model_input_ptr(0);
  // copy data to inputs
  model_invoke();
  int8_t *outputs = (int8_t *)model_output_ptr(0);
  // print outputs.
}
