#include "1.cpp.h"
#include<stdio.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>

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


int main(int argc, char *argv[])
{
  (void)load_binary_file(argv[1], params_content, MAX_PARAMS_SIZE);

  if(model_init(params_content)){
    printf("Error!\n");
  }

  xt::xarray<int8_t> input = xt::load_npy<int8_t>("input.npy");
  int8_t *in = model_input(0)->data.int8;
  for (int i=0;i<model_input_size(0);++i) {
    in[i] = input[i];
  }
  printf("\n");

  model_invoke();

  for(int n=0; n< model_outputs(); ++n) {
    int8_t *out = model_output(n)->data.int8;
     for (int i=0;i<model_output_size(n);++i){
	if (out[i] != -128) {
            //printf("%d %d\n",i, (int)out[i]);
        }
       //printf("%d,",(int)out[i]);
     }
    //printf(" checksum : %d\n", (int)checksum_calc((char*)out, model_output_size(n)));
  }

  return 0;
}