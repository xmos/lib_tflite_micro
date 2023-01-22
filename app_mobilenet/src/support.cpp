#include <platform.h>
#include <stdint.h>
#include <stdio.h>
#include <xcore/channel.h>

#include "model.tflite.h"

#define TEST_INPUT_SIZE  (224*224*3)
#define TEST_OUTPUT_SIZE0  (1000)

static int once = 1;

void run(unsigned x, unsigned flash_data) {

    if (once) {
    printf("\nModel init");
    model_init((void*)flash_data);
    printf("\nModel init done");
    once = 0;
    }

    printf("\nModel input");
    int8_t *p = model_input(0)->data.int8;
    for(int i = 0 ; i < TEST_INPUT_SIZE; i++) {
        p[i] = chanend_in_byte(x);
    }
    printf("\nModel input done");

    printf("\nModel invoke");
    model_invoke();
    printf("\nModel invoke done");

    int8_t *o0 = model_output(0)->data.int8;
    for(int i = 0; i < TEST_OUTPUT_SIZE0; i++) {
        chanend_out_byte(x, o0[i]);
    }

    chanend_out_end_token(x);
    chanend_check_end_token(x);
}


extern "C" {
    void inferencer(unsigned x, unsigned flash_data) {
        run(x, flash_data);
    }
}