#include "run.h"
#include "detect.cpp.h"
#include "rcgn.cpp.h"
#include <stdio.h>

int run_detect(dummy_t * d, void *flash_data)
{
  printf("in run detect\n");

  detect_init(flash_data);

  printf("after detect init\n");
 
  int8_t *in = detect_input(0)->data.int8;
  for (int i=0;i<detect_input_size(0);++i) {
    in[i]=110;
  }

  printf("before detect invoke\n");
	
  detect_invoke();

  printf("after detect invoke\n");
	
  int8_t *out = detect_output(0)->data.int8;
  for (int i=0;i<detect_output_size(0);++i)
  	printf("%d,",(int)out[i]);
  printf("\n");
  return 0;
}

int run_rcgn()
{
  printf("in run rcgn\n");

  rcgn_init();

  printf("after init rcgn\n");
 
  int8_t *in = rcgn_input(0)->data.int8;
  for (int i=0;i<rcgn_input_size(0);++i) {
    in[i]=110;
  }

  printf("before invoke rcgn\n");
	
  rcgn_invoke();

  printf("after invoke rcgn\n");
	
  int8_t *out = rcgn_output(0)->data.int8;
  for (int i=0;i<rcgn_output_size(0);++i)
  	printf("%d,",(int)out[i]);
  printf("\n");
  return 0;
}