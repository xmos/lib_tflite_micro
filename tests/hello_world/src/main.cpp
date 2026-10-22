#include <stdio.h>
#include <assert.h>
#include <stdint.h>

// This code runs in a loop the hello world tflite model from:
// lib_tflite_micro/submodules/tflite-micro/tensorflow/lite/micro/examples/hello_world

#include "hello_world_int8_opt.tflite.h"

#define INPUT_FILE "input.bin"
#define OUTPUT_FILE "output.bin"

void generic_run_model(void)
{
    TfLiteStatus status = model_init(NULL);
    assert(status == kTfLiteOk);
    int8_t *inputs = (int8_t *)model_input_ptr(0);
    int8_t *outputs = (int8_t *)model_output_ptr(0);
    size_t input_size = model_input_size(0);
    size_t output_size = model_output_size(0);
    assert(input_size == 1);
    assert(output_size == 1);

    FILE *input_file = fopen(INPUT_FILE, "rb");
    FILE *output_file = fopen(OUTPUT_FILE, "wb");
    assert(input_file != NULL);
    assert(output_file != NULL);

    // read, invoke, write prediction
    while (fread(inputs, 1, input_size, input_file) == input_size) {
        status = model_invoke();
        assert(status == kTfLiteOk);
        size_t bytes_written = fwrite(outputs, 1, output_size, output_file);
        assert(bytes_written == output_size);
    }
    assert(!ferror(input_file));

    int file_status = fclose(input_file);
    assert(file_status == 0);
    file_status = fclose(output_file);
    assert(file_status == 0);
}

int main(void) {
    generic_run_model();
    return 0;
}
