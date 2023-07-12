#include <platform.h>
#include <stdint.h>
#include <stdio.h>
#include <xcore/channel.h>

#include "xcore_optimised_mobilenetV2_input_160x160x3_alpha_1.tflite.h"

unsigned char checksum_calc(char *data, unsigned int length) {
  static char sum;
  static char *end;
  sum = 0;
  end = data + length;

  do {
    sum -= *data++;
  } while (data != end);
  return sum;
}

void init1(unsigned flash_data) {
  printf("\nModel1 init");
  model_init((void *)flash_data);
  printf("\nModel1 init done");
}

void run1() {
  printf("\nModel1 input");
  int8_t *p = model_input(0)->data.int8;
  // input is simply filled with -128 to 127 repeatedly
  int k = -128;
  for (int i = 0; i < model_input_size(0); ++i) {
    if (k == 128) {
      k = -128;
    }
    p[i] = k;
    k++;
  }
  printf("\nModel1 input done");

  printf("\nModel1 invoke");
  model_invoke();
  printf("\nModel1 invoke done");

  for (int n = 0; n < model_outputs(); ++n) {
    int8_t *out = model_output(n)->data.int8;
    for (int i = 0; i < model_output_size(n); ++i) {
      // printf("%d,",(int)out[i]);
    }
    printf("\nchecksum : %d\n\n",
           (int)checksum_calc((char *)out, model_output_size(n)));
  }
}

extern "C" {
void model1_init(unsigned flash_data) { init1(flash_data); }

void inference1() { run1(); }
}
