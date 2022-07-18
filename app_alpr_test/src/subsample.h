#include <stdint.h>

#define SUBSAMPLE_MAX_WINDOW_SIZE  5
#define SUBSAMPLE_MAX_OUTPUT_WIDTH   160
#define SUBSAMPLE_MAX_OUTPUT_HEIGHT   160

void subsample_y(int8_t *outp,
                 int8_t *inp0,
                 int8_t *inp1,
                 int8_t *inp2,
                 int8_t *inp3,
                 int8_t *inp4,
                 int8_t coefficients[], int nox);

void subsample_x(int8_t outp[3][SUBSAMPLE_MAX_OUTPUT_WIDTH],
                 uint8_t line[],
                 int8_t coefficients[], uint32_t strides[], int nox);

void build_y_coefficients_strides(int8_t y_coefficients[16*SUBSAMPLE_MAX_OUTPUT_HEIGHT*SUBSAMPLE_MAX_WINDOW_SIZE],
                                  uint32_t strides[SUBSAMPLE_MAX_OUTPUT_HEIGHT],
                                  int start_y, int end_y, int points);


void build_x_coefficients_strides(int8_t x_coefficients[32*SUBSAMPLE_MAX_OUTPUT_WIDTH*3],
                                  uint32_t strides[SUBSAMPLE_MAX_OUTPUT_WIDTH],
                                  int start_x, int end_x, int points);


