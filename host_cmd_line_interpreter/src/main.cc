// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1
#include "lib_tflite_micro/api/inference_engine.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define MAX_MODEL_CONTENT_SIZE 5000000
#define MAX_PARAMS_SIZE 5000000
#define MAX_ARENA_SIZE 50000000

uint32_t model_content[MAX_MODEL_CONTENT_SIZE / sizeof(uint32_t)];
uint32_t params_content[MAX_MODEL_CONTENT_SIZE / sizeof(uint32_t)];
uint32_t arena[MAX_ARENA_SIZE / sizeof(uint32_t)];

struct tflite_micro_objects s0;

void inference_engine_initialize(inference_engine_t *ie) {
  s0.interpreter = nullptr;
  auto *resolver = inference_engine_initialize(
      ie, arena, MAX_ARENA_SIZE, nullptr, 0, &s0);
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
  resolver->AddStridedSlice();
  tflite::ops::micro::xcore::RegisterXCOps(resolver);
}

static int load_binary_file(const char *filename, uint32_t *content,
                            size_t size) {
  if (strcmp(filename, "-") == 0) {
    return 0;
  }
  FILE *fd = fopen(filename, "rb");
  if (fd == NULL) {
    fprintf(stderr, "Cannot read model/param file %s\n", filename);
    exit(1);
  }
  int s = fread(content, 1, size, fd);
  fclose(fd);

  return s;
}

static int load_input(const char *filename, uint32_t *input, size_t esize) {
  FILE *fd = fopen(filename, "rb");
  if (fd == NULL) {
    fprintf(stderr, "Cannot read input file %s\n", filename);
    exit(1);
  }
  int s = fread(input, 1, esize, fd);
  fclose(fd);

  if (s != esize) {
    printf("ERROR: Incorrect input file '%s'. Expected %zu bytes got %d.\n",
           filename, esize, s);
  }
  return s;
}

static int save_output(const char *filename, const uint32_t *output,
                       size_t osize) {
  FILE *fd = fopen(filename, "wb");
  if (fd == NULL) {
    fprintf(stderr, "Cannot write output file %s\n", filename);
    exit(1);
  }
  fwrite(output, 1, osize, fd);
  fclose(fd);

  return 1;
}

int main(int argc, char *argv[]) {
  int carg = 3;
  if (argc < 5) {
    fprintf(stderr, "Usage\n");
    fprintf(stderr,
            "   %s: model.tflite model.params input-file output-file; or\n",
            argv[0]);
    fprintf(stderr,
            "   %s: model.tflite model.params -i input-files ... -o "
            "output-files ...\n",
            argv[0]);
    return -1;
  }
  char *model_filename = argv[1];

  inference_engine_t ie;
  inference_engine_initialize(&ie);

  // load model
  size_t model_size = load_binary_file(model_filename, model_content,
                                       MAX_MODEL_CONTENT_SIZE);

  char *params_filename = argv[2];
  (void)load_binary_file(params_filename, params_content, MAX_PARAMS_SIZE);

  inference_engine_unload_model(&ie);
  int error = inference_engine_load_model(&ie, model_size, model_content,
                                          params_content);

  if (strcmp(argv[carg], "-i") == 0) {

    int tensor_num = 0;
    while(strcmp(argv[++carg], "-o") != 0 && carg < argc) {
        load_input(argv[carg], ie.input_buffers[tensor_num], ie.input_sizes[tensor_num]);
        tensor_num++;
    }
    interp_invoke_par_5(&ie);
    printf("%d\n", ie.arena_needed_bytes);
    tensor_num = 0;
    while(++carg < argc) {
        save_output(argv[carg], ie.output_buffers[tensor_num], ie.output_sizes[tensor_num]);
        tensor_num++;
    }
  } else {
    char *input_filename = argv[carg];
    char *output_filename = argv[carg + 1];
    load_input(input_filename, ie.input_buffers[0], ie.input_sizes[0]);
    interp_invoke_par_5(&ie);
    save_output(output_filename, ie.output_buffers[0], ie.output_sizes[0]);
  }

  return 0;
}
