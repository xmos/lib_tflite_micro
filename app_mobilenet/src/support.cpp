#include <platform.h>
#include <stdint.h>
#include <stdio.h>
#include <xcore/channel.h>

#include "model.tflite.h"

#define TEST_INPUT_SIZE  (256*192*3)
#define TEST_OUTPUT_SIZE0  (80)
#define TEST_OUTPUT_SIZE1  (20)
#define TEST_OUTPUT_SIZE2  (20)
#define TEST_OUTPUT_SIZE3  (1)

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

    int32_t *o0 = model_output(0)->data.i32;
    for(int i = 0; i < TEST_OUTPUT_SIZE0; i++) {
        chanend_out_word(x, o0[i]);
    }

    int32_t *o1 = model_output(1)->data.i32;
    for(int i = 0; i < TEST_OUTPUT_SIZE1; i++) {
        chanend_out_word(x, o1[i]);
    }

    int8_t *o2 = model_output(2)->data.int8;
    for(int i = 0; i < TEST_OUTPUT_SIZE2; i++) {
        chanend_out_byte(x, o2[i]);
    }

    int32_t *o3 = model_output(3)->data.i32;
    for(int i = 0; i < TEST_OUTPUT_SIZE3; i++) {
        chanend_out_word(x, o3[i]);
    }

    chanend_out_end_token(x);
    chanend_check_end_token(x);
}


extern "C" {
    void inferencer(unsigned x, unsigned flash_data) {
        run(x, flash_data);
    }
}