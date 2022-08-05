#include "model.cpp.h"
#include<stdio.h>

#include <xtensor/xarray.hpp>
#include <xtensor/xnpy.hpp>

int main(int argc, char *argv[])
{
  if(model_init()){
    printf("Error!\n");
  }

  char *input_filename = argv[1];
 
  xt::xarray<int8_t> input = xt::load_npy<int8_t>(input_filename);
  int8_t *in = model_input(0)->data.int8;
  for (int i=0;i<model_input_size(0);++i) {
    printf("%d,",(int)input.flat(i));
    in[i] = input.flat(i);
  }
  printf("\n");

  model_invoke();

  for(int n=0; n< model_outputs(); ++n) {
    int8_t *out = model_output(n)->data.int8;
    xt::xarray<int8_t> output;
    output.resize({model_output_size(n)});

    for (int i=0;i<model_output_size(n);++i){
      printf("%d,",(int)out[i]);
      output[i] = out[i];
    }
    xt::dump_npy(std::to_string(n) + ".npy", output);
    printf("\n");
  }
  return 0;
}
