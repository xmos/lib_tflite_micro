#include "model.tflite.h"
#include<stdio.h>

// #include <xtensor/xarray.hpp>
// #include <xtensor/xnpy.hpp>

unsigned char checksum_calc(char *data, unsigned int length)
{
  static char sum;
  static char * end;
  sum = 0;
  end = data + length;

  do
  {
      sum -= *data++;
  } while (data != end);
  return sum;
}

#define MAX_PARAMS_SIZE 5000000
#define MAX_MODEL_CONTENT_SIZE 5000000
static int load_binary_file(const char *filename, uint32_t *content,
                            size_t size) {
  FILE *fd = fopen(filename, "rb");
  if (fd == NULL) {
    fprintf(stderr, "Cannot read model/param file %s\n", filename);
  }
  int s = fread(content, 1, size, fd);
  fclose(fd);

  return s;
}
uint32_t params_content[MAX_MODEL_CONTENT_SIZE / sizeof(uint32_t)];

#define I16

int main(int argc, char *argv[])
{
  (void)load_binary_file(argv[1], params_content, MAX_PARAMS_SIZE);

  if(model_init(params_content)){
    printf("Error!\n");
  }

  //xt::xarray<int8_t> input = xt::load_npy<int8_t>("input.npy");
  
  for(int n=0; n< model_inputs(); ++n) {
    //int32_t *in = model_input(n)->data.i32;
    #ifdef I16
    int16_t *in = model_input(n)->data.i16;
    int size = model_input_size(n)/2;
    int k = -32768;
    for (int i=0;i<size;++i) {
      if (k >= 32767) {
        k = -32768;
      }
      in[i] = k;//input[i];
      k = k + 5000;
    }
    #else
    int8_t *in = model_input(n)->data.int8;
    int size = model_input_size(n);
    int k = -128;
    for (int i=0;i<size;++i) {
      if (k >= 128) {
        k = -128;
      }
      in[i] = k;//input[i];
      k = k + 3;
    }
    #endif
  }
  printf("\n");

  model_invoke();

  for(int n=0; n< model_outputs(); ++n) {
    //int32_t *out = model_output(n)->data.i32;
    #ifdef I16
    int16_t *out = model_output(n)->data.i16;
    int size = model_output_size(n)/2;
    #else
    int8_t *out = model_output(n)->data.int8;
    int size = model_output_size(n);
    #endif
     for (int i=0;i<size;++i){
       printf("%d,",(int)out[i]);
     }
    printf("\nchecksum : %d\n\n", (int)checksum_calc((char*)out, model_output_size(n)));
  }

  return 0;
}
