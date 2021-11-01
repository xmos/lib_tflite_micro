// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "inference_engine.h"

#define MAX_MODEL_CONTENT_SIZE 5000000

uint32_t model_content[MAX_MODEL_CONTENT_SIZE / sizeof(uint32_t)];

struct tflite_micro_objects s0;

void inference_engine_initialize(inference_engine_t *ie) {
    auto *resolver = inference_engine_initialize(ie,
                                                 model_content, MAX_MODEL_CONTENT_SIZE,
                                                 nullptr,  0,
                                                 &s0);
    resolver->AddDequantize();
    resolver->AddSoftmax();
    resolver->AddMean();
    resolver->AddPad();
    resolver->AddReshape();
    resolver->AddConcatenation();
    resolver->AddAdd();
    resolver->AddLogistic();
    resolver->AddConv2D();
    resolver->AddQuantize();
    resolver->AddDepthwiseConv2D();
    resolver->AddCustom(tflite::ops::micro::xcore::Conv2D_V2_OpCode,
                        tflite::ops::micro::xcore::Register_Conv2D_V2());
    resolver->AddCustom(tflite::ops::micro::xcore::Load_Flash_OpCode,
                        tflite::ops::micro::xcore::Register_LoadFromFlash());

}



static int load_model(const char *filename, uint32_t *content,
                      size_t size) {
    FILE *fd = fopen(filename, "rb");
    int s = fread(content, 1, size, fd);
    fclose(fd);

    return s;
}

static int load_input(const char *filename, uint32_t *input,
                      size_t esize) {
    FILE *fd = fopen(filename, "rb");
    int s = fread(input, 1, esize, fd);
    fclose(fd);

    if (s != esize) {
        printf("ERROR: Incorrect input file size. Expected %zu bytes.\n", esize);
    }
    return s;
}

static int save_output(const char *filename, const uint32_t *output,
                       size_t osize) {
    FILE *fd = fopen(filename, "wb");
    fwrite(output, 1, osize, fd);
    fclose(fd);

    return 1;
}

int main(int argc, char *argv[]) {
    int carg = 2;
    if (argc < 4) {
        fprintf(stderr, "Usage\n");
        fprintf(stderr, "   %s: model.tflite input-file output-file; or\n", argv[0]);
        fprintf(stderr, "   %s: model.tflite -i input-files ... -o output-files ...\n", argv[0]);
        return -1;
    }

    char *model_filename = argv[1];

    inference_engine_t ie;
    inference_engine_initialize(&ie);

    // load model
    size_t model_size = load_model(model_filename, ie.model_data_tensor_arena, MAX_MODEL_CONTENT_SIZE);


    inference_engine_unload_model(&ie);
    int error = inference_engine_load_model(&ie, model_size, ie.model_data_tensor_arena, 0); // TODO: c_flash!!

    if (strcmp(argv[carg], "-i") == 0) {
        int tensor_num = 0;
        while(strcmp(argv[++carg], "-o") != 0 && carg < argc) {
            load_input(argv[carg], ie.input_buffers[tensor_num], ie.input_sizes[tensor_num]);
            tensor_num++;
        }
        interp_invoke(&ie);
        tensor_num = 0;
        while(++carg < argc) {
            save_output(argv[carg], ie.output_buffers[tensor_num], ie.output_sizes[tensor_num]);
            tensor_num++;
        }
    } else {
        char *input_filename = argv[2];
        char *output_filename = argv[3];
        load_input(input_filename, ie.input_buffers[0], ie.input_sizes[0]);
        interp_invoke(&ie);
        save_output(output_filename, ie.output_buffers[0], ie.output_sizes[0]);
    }

    return 0;
}
