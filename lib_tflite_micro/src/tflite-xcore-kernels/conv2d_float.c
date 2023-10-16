#include "conv2d_float.h"
#include <assert.h>

int xc_fc_float_ref(float *outputs, float *inputs, float *kernels,
                    int out_features, int input_features) {
  int cnt = 0;
  for (int f = 0; f < out_features; f++) {
    int output_index = f;
    float acc = 0;
    for (int kf = 0; kf < input_features; kf++) {
      int input_index = kf;
      int kernel_index = f * input_features + kf;
      acc += inputs[input_index] * kernels[kernel_index];
      cnt++;
    }
    outputs[output_index] = acc;
  }
  return cnt;
}

#ifndef NN_USE_REF
int xc_fc_float_opt(float *outputs, float *inputs, float *kernels,
                    int out_features, int input_features, int out_f_start,
                    int out_f_end) {
  int cnt = 0;
  for (int f = out_f_start; f < out_f_end; f++) {
    int output_index = f;
    float acc = 0;
    for (int kf = 0; kf < input_features; kf++) {
      int input_index = kf;
      int kernel_index = f * input_features + kf;
      float in1 = inputs[input_index];
      float in2 = kernels[kernel_index];
      asm volatile("fmacc %0, %1, %2, %3"
                   : "=r"(acc)
                   : "r"(acc), "r"(in1), "r"(in2));
      cnt++;
    }
    outputs[output_index] = acc;
  }
  return cnt;
}
#endif

#define KW 3
#define KH 2
#define H_STRIDE 2

int xc_conv2d_float_kw5xh2_stride_w3_ref(float *outputs, float *inputs,
                                         float *kernels, float *biases,
                                         int out_w, int out_h, int out_depth,
                                         int input_w, int input_h,
                                         int input_depth) {
  int cnt = 0;
  for (int x = 0; x < out_w; x++) {
    for (int y = 0; y < out_h; y++) {
      for (int d = 0; d < out_depth; d++) {
        int output_index = (x * out_h + y) * out_depth + d;
        float acc = biases[d];
        for (int kx = 0; kx < KW; kx++) {
          for (int ky = 0; ky < KH; ky++) {
            for (int kd = 0; kd < input_depth; kd++) {
              int input_index =
                  ((x * H_STRIDE + kx) * input_h + (y + ky)) * input_depth + kd;
              int kernel_index = ((d * KW + kx) * KH + ky) * input_depth + kd;
              acc += inputs[input_index] * kernels[kernel_index];
              cnt++;
            }
          }
        }
        outputs[output_index] = acc;
      }
    }
  }
  return cnt;
}

#ifndef NN_USE_REF
void xc_conv2d_float_kw5xh2_stride_w3_opt(float *outputs, float *inputs,
                                          float *kernels, float *biases,
                                          int out_w, int out_h, int out_depth,
                                          int input_w, int input_h,
                                          int input_depth, int out_depth_start,
                                          int out_depth_end) {
  for (int x = 0; x < out_w; x++) {
    for (int y = 0; y < out_h; y++) {
      for (int d = out_depth_start; d < out_depth_end; d++) {
        int output_index = (x * out_h + y) * out_depth + d;
        float acc = biases[d];
        if (input_depth == 1) {
#pragma clang loop unroll(full)
          for (int kx = 0; kx < KW; kx++) {
#pragma clang loop unroll(full)
            for (int ky = 0; ky < KH; ky++) {
              int input_index =
                  ((x * H_STRIDE + kx) * input_h + (y + ky)) * input_depth;
              int kernel_index = ((d * KW + kx) * KH + ky) * input_depth;
              float in1 = inputs[input_index];
              float in2 = kernels[kernel_index];
              asm volatile("fmacc %0, %1, %2, %3"
                           : "=r"(acc)
                           : "r"(acc), "r"(in1), "r"(in2));
            }
          }
        } else if (input_depth == 2) {
#pragma clang loop unroll(full)
          for (int kx = 0; kx < KW; kx++) {
#pragma clang loop unroll(full)
            for (int ky = 0; ky < KH; ky++) {
              int input_index =
                  ((x * H_STRIDE + kx) * input_h + (y + ky)) * input_depth;
              int kernel_index = ((d * KW + kx) * KH + ky) * input_depth;
              float in1 = inputs[input_index];
              float in2 = kernels[kernel_index];
              asm volatile("fmacc %0, %1, %2, %3"
                           : "=r"(acc)
                           : "r"(acc), "r"(in1), "r"(in2));
            }
          }
        } else if (input_depth == 4) {
          for (int kx = 0; kx < KW; kx++) {
            for (int ky = 0; ky < KH; ky++) {
#pragma clang loop unroll(full)
              for (int kd = 0; kd < 4; kd++) {
                int input_index =
                    ((x * H_STRIDE + kx) * input_h + (y + ky)) * input_depth +
                    kd;
                int kernel_index = ((d * KW + kx) * KH + ky) * input_depth + kd;
                float in1 = inputs[input_index];
                float in2 = kernels[kernel_index];
                asm volatile("fmacc %0, %1, %2, %3"
                             : "=r"(acc)
                             : "r"(acc), "r"(in1), "r"(in2));
              }
            }
          }
        } else if (input_depth == 8) {
          for (int kx = 0; kx < KW; kx++) {
            for (int ky = 0; ky < KH; ky++) {
#pragma clang loop unroll(full)
              for (int kd = 0; kd < 8; kd++) {
                int input_index =
                    ((x * H_STRIDE + kx) * input_h + (y + ky)) * input_depth +
                    kd;
                int kernel_index = ((d * KW + kx) * KH + ky) * input_depth + kd;
                float in1 = inputs[input_index];
                float in2 = kernels[kernel_index];
                asm volatile("fmacc %0, %1, %2, %3"
                             : "=r"(acc)
                             : "r"(acc), "r"(in1), "r"(in2));
              }
            }
          }
        } else if (input_depth == 16) {
          for (int kx = 0; kx < KW; kx++) {
            for (int ky = 0; ky < KH; ky++) {
#pragma clang loop unroll_count(8)
              for (int kd = 0; kd < 16; kd++) {
                int input_index =
                    ((x * H_STRIDE + kx) * input_h + (y + ky)) * input_depth +
                    kd;
                int kernel_index = ((d * KW + kx) * KH + ky) * input_depth + kd;
                float in1 = inputs[input_index];
                float in2 = kernels[kernel_index];
                asm volatile("fmacc %0, %1, %2, %3"
                             : "=r"(acc)
                             : "r"(acc), "r"(in1), "r"(in2));
              }
            }
          }
        } else {
          assert(0);
        }
        outputs[output_index] = acc;
      }
    }
  }
}
#endif

#define H_TR_STRIDE 2

int xc_transpose_conv2d_float_kw5xh2_stride_h3_ref(
    float *outputs, float *inputs, float *kernels, float *biases, int out_w,
    int out_h, int out_depth, int input_w, int input_h, int input_depth) {
  int cnt = 0;
  for (int x = 0; x < out_w; x++) {
    for (int y = 0; y < out_h; y++) {
      for (int d = 0; d < out_depth; d++) {
        int output_index = (x * out_h + y) * out_depth + d;
        outputs[output_index] = biases[d];
      }
    }
  }
  for (int x = 0; x < input_w; x++) {
    for (int y = 0; y < input_h; y++) {
      for (int d = 0; d < out_depth; d++) {
        for (int kx = 0; kx < KW; kx++) {
          for (int ky = 0; ky < KH; ky++) {
            int output_index =
                ((x * H_TR_STRIDE + kx) * out_h + (y + ky)) * out_depth + d;
            float acc = outputs[output_index];
            for (int kd = 0; kd < input_depth; kd++) {
              int input_index = ((x)*input_h + (y)) * input_depth + kd;
              int kernel_index = ((d * KW + kx) * KH + ky) * input_depth + kd;
              acc += inputs[input_index] * kernels[kernel_index];
              cnt++;
            }
            outputs[output_index] = acc;
          }
        }
      }
    }
  }
  return cnt;
}

#ifndef NN_USE_REF
void xc_transpose_conv2d_float_kw5xh2_stride_h3_opt(
    float *outputs, float *inputs, float *kernels, float *biases, int out_w,
    int out_h, int out_depth, int input_w, int input_h, int input_depth,
    int out_depth_start, int out_depth_end) {
  for (int x = 0; x < out_w; x++) {
    for (int y = 0; y < out_h; y++) {
      for (int d = out_depth_start; d < out_depth_end; d++) {
        int output_index = (x * out_h + y) * out_depth + d;
        outputs[output_index] = biases[d];
      }
    }
  }
  for (int x = 0; x < input_w; x++) {
    for (int y = 0; y < input_h; y++) {
      for (int d = out_depth_start; d < out_depth_end; d++) {
        if (input_depth == 4) {
          for (int kx = 0; kx < KW; kx++) {
            for (int ky = 0; ky < KH; ky++) {
              int output_index =
                  ((x * H_TR_STRIDE + kx) * out_h + (y + ky)) * out_depth + d;
              float acc = outputs[output_index];
#pragma clang loop unroll_count(4)
              for (int kd = 0; kd < 4; kd++) {
                int input_index = ((x)*input_h + (y)) * input_depth + kd;
                int kernel_index = ((d * KW + kx) * KH + ky) * input_depth + kd;
                float in1 = inputs[input_index];
                float in2 = kernels[kernel_index];
                asm volatile("fmacc %0, %1, %2, %3"
                             : "=r"(acc)
                             : "r"(acc), "r"(in1), "r"(in2));
              }
              outputs[output_index] = acc;
            }
          }
        } else if (input_depth == 8) {
          for (int kx = 0; kx < KW; kx++) {
            for (int ky = 0; ky < KH; ky++) {
              int output_index =
                  ((x * H_TR_STRIDE + kx) * out_h + (y + ky)) * out_depth + d;
              float acc = outputs[output_index];
#pragma clang loop unroll_count(8)
              for (int kd = 0; kd < 8; kd++) {
                int input_index = ((x)*input_h + (y)) * input_depth + kd;
                int kernel_index = ((d * KW + kx) * KH + ky) * input_depth + kd;
                float in1 = inputs[input_index];
                float in2 = kernels[kernel_index];
                asm volatile("fmacc %0, %1, %2, %3"
                             : "=r"(acc)
                             : "r"(acc), "r"(in1), "r"(in2));
              }
              outputs[output_index] = acc;
            }
          }
        } else if (input_depth == 16) {
          for (int kx = 0; kx < KW; kx++) {
            for (int ky = 0; ky < KH; ky++) {
              int output_index =
                  ((x * H_TR_STRIDE + kx) * out_h + (y + ky)) * out_depth + d;
              float acc = outputs[output_index];
#pragma clang loop unroll_count(8)
              for (int kd = 0; kd < 16; kd++) {
                int input_index = ((x)*input_h + (y)) * input_depth + kd;
                int kernel_index = ((d * KW + kx) * KH + ky) * input_depth + kd;
                float in1 = inputs[input_index];
                float in2 = kernels[kernel_index];
                asm volatile("fmacc %0, %1, %2, %3"
                             : "=r"(acc)
                             : "r"(acc), "r"(in1), "r"(in2));
              }
              outputs[output_index] = acc;
            }
          }
        } else if (input_depth == 32) {
          for (int kx = 0; kx < KW; kx++) {
            for (int ky = 0; ky < KH; ky++) {
              int output_index =
                  ((x * H_TR_STRIDE + kx) * out_h + (y + ky)) * out_depth + d;
              float acc = outputs[output_index];
#pragma clang loop unroll_count(8)
              for (int kd = 0; kd < 32; kd++) {
                int input_index = ((x)*input_h + (y)) * input_depth + kd;
                int kernel_index = ((d * KW + kx) * KH + ky) * input_depth + kd;
                float in1 = inputs[input_index];
                float in2 = kernels[kernel_index];
                asm volatile("fmacc %0, %1, %2, %3"
                             : "=r"(acc)
                             : "r"(acc), "r"(in1), "r"(in2));
              }
              outputs[output_index] = acc;
            }
          }
        } else if (input_depth == 64) {
          for (int kx = 0; kx < KW; kx++) {
            for (int ky = 0; ky < KH; ky++) {
              int output_index =
                  ((x * H_TR_STRIDE + kx) * out_h + (y + ky)) * out_depth + d;
              float acc = outputs[output_index];
#pragma clang loop unroll_count(8)
              for (int kd = 0; kd < 64; kd++) {
                int input_index = ((x)*input_h + (y)) * input_depth + kd;
                int kernel_index = ((d * KW + kx) * KH + ky) * input_depth + kd;
                float in1 = inputs[input_index];
                float in2 = kernels[kernel_index];
                asm volatile("fmacc %0, %1, %2, %3"
                             : "=r"(acc)
                             : "r"(acc), "r"(in1), "r"(in2));
              }
              outputs[output_index] = acc;
            }
          }
        } else {
          assert(0);
        }
      }
    }
  }
}
#endif
