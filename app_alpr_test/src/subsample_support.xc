#include <stdio.h>
#include <print.h>
#include <stdint.h>
#include <string.h>
#include "subsample.h"

void transform_line(int8_t outp[3][160], uint8_t line[]) {
    for(int ox = 0; ox < 160; ox ++) {
        int x = ox;
        int Y = line[2*x];
        int UV0 = line[x == 159 ? 2*x - 3 : 2*x+1];
        int UV1 = line[x == 0 ? 2*x + 3 : 2*x-1];
        int U, V;
        if ((x & 1) == 1) {
            U = UV0; V = UV1;
        } else {
            U = UV1; V = UV0;
        }
        Y -= 128;
        U -= 128;
        V -= 128;
        int R = Y + ((          292 * V) >> 8);
        int G = Y - ((100 * U + 148 * V) >> 8);
        int B = Y + ((520 * U          ) >> 8);
        if (R < -128) R = -128; if (R > 127) R = 127;
        if (G < -128) G = -128; if (G > 127) G = 127;
        if (B < -128) B = -128; if (B > 127) B = 127;
        outp[0][x] = R;
        outp[1][x] = G;
        outp[2][x] = B;
    }
}

int16_t shift_8[16] = {
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
};

void p(int8_t x[32]) {
    for(int i = 0; i < 32; i++) {
        printf("%02x ", x[i]&0xff);
    }
    printf("\n");
}

#pragma unsafe arrays

extern void subsample_x_asm(int8_t outp[3][SUBSAMPLE_MAX_OUTPUT_WIDTH], uint8_t line[], int8_t coefficients[], uint32_t strides[], int ox, int indexc);
void transform_part_line_vpu(int8_t outp[3][SUBSAMPLE_MAX_OUTPUT_WIDTH], uint8_t line[], int8_t coefficients[], uint32_t strides[], int ox, int indexc) {
#pragma loop unroll(3)
    for(int rgb = 0; rgb < 3; rgb++) {
        asm volatile("vclrdr");
#pragma loop unroll(16)
        for(int j = 0; j < 16; j++) {
            asm volatile("vldc %0[0]" :: "r" (&coefficients[indexc]));
            indexc+=32;
            asm volatile("vlmaccr %0[0]" :: "r" (&line[strides[ox+j]]));
        }
        asm volatile("vlsat %0[0]" :: "r" (shift_8));
        asm volatile("vdepth8");
        asm volatile("vstrpv %0[0], %1" :: "r" (&outp[rgb][ox]), "r" (0xFFFF));
    }
}

#pragma unsafe arrays
void subsample_x(int8_t outp[3][SUBSAMPLE_MAX_OUTPUT_WIDTH], uint8_t line[], int8_t coefficients[], uint32_t strides[], int nox) {
    int indexc = 0;
    for(int ox = 0; ox < nox; ox +=16) {
        subsample_x_asm(outp, line, coefficients, strides, ox, indexc);
        indexc += 16 * 3 * 32;
    }
}

void transform_line_y(int8_t outp[SUBSAMPLE_MAX_OUTPUT_WIDTH][3],
                      int8_t inp0[3][SUBSAMPLE_MAX_OUTPUT_WIDTH],
                      int8_t inp1[3][SUBSAMPLE_MAX_OUTPUT_WIDTH],
                      int8_t inp2[3][SUBSAMPLE_MAX_OUTPUT_WIDTH],
                      int8_t inp3[3][SUBSAMPLE_MAX_OUTPUT_WIDTH],
                      int8_t inp4[3][SUBSAMPLE_MAX_OUTPUT_WIDTH],
                      int8_t coefficients[], int nox) {
    int t0, t1;
    asm volatile ("gettime %0" : "=r" (t0));
    for(int ox = 0; ox < nox; ox ++) {
        for(int rgb = 0; rgb < 3; rgb++) {
            int sum = 0;
            sum += coefficients[0*16] * inp0[rgb][ox];
            sum += coefficients[1*16] * inp1[rgb][ox];
            sum += coefficients[2*16] * inp2[rgb][ox];
            sum += coefficients[3*16] * inp3[rgb][ox];
            sum += coefficients[4*16] * inp4[rgb][ox];
            sum = sum >> 6;
            if (sum > 127) sum = 127;
            if (sum < -128) sum = -128;
            outp[ox][rgb] = sum;
            if (0 && ox == 0) {
                printf("%d %d %d %d %d     %d %d %d %d %d", coefficients[0*16], coefficients[1*16], coefficients[2*16], coefficients[3*16], coefficients[4*16]
                       , inp0[rgb][0], inp1[rgb][0], inp2[rgb][0], inp3[rgb][0], inp4[rgb][4]);
                printf(" & %08x  %d\n", &outp[ox][0], outp[ox][rgb]);
            }
        }
    }
    asm volatile("gettime %0" : "=r" (t1));
}

void subsample_y_rgb(int8_t outp[SUBSAMPLE_MAX_OUTPUT_WIDTH][3],
                              int8_t inp0[3][SUBSAMPLE_MAX_OUTPUT_WIDTH],
                              int8_t inp1[3][SUBSAMPLE_MAX_OUTPUT_WIDTH],
                              int8_t inp2[3][SUBSAMPLE_MAX_OUTPUT_WIDTH],
                              int8_t inp3[3][SUBSAMPLE_MAX_OUTPUT_WIDTH],
                              int8_t inp4[3][SUBSAMPLE_MAX_OUTPUT_WIDTH],
                              int8_t coefficients[], int nox, int rgb) {
    int8_t tmp[32];
    for(int ox = 0; ox < nox; ox +=16) {
        asm volatile ("vclrdr");
        asm volatile ("vldc   %0[0]" :: "r" (&inp0[rgb][ox]));
        asm volatile ("vlmacc %0[0]" :: "r" (&coefficients[0*16]));
        asm volatile ("vldc   %0[0]" :: "r" (&inp1[rgb][ox]));
        asm volatile ("vlmacc %0[0]" :: "r" (&coefficients[1*16]));
        asm volatile ("vldc   %0[0]" :: "r" (&inp2[rgb][ox]));
        asm volatile ("vlmacc %0[0]" :: "r" (&coefficients[2*16]));
        asm volatile ("vldc   %0[0]" :: "r" (&inp3[rgb][ox]));
        asm volatile ("vlmacc %0[0]" :: "r" (&coefficients[3*16]));
        asm volatile ("vldc   %0[0]" :: "r" (&inp4[rgb][ox]));
        asm volatile ("vlmacc %0[0]" :: "r" (&coefficients[4*16]));
        asm volatile ("vlsat  %0[0]" :: "r" (&shift_8));
        asm volatile ("vdepth8" :: "r" (&shift_8));
        asm volatile ("vstr   %0[0]" :: "r" (&tmp));
        for(int i = 0; i < 16; i++) {
            outp[ox+i][rgb] = tmp[i];
        }
    }
}


extern void subsample_y_asm(int8_t *outp,
                            int8_t *inp0,
                            int8_t *inp1,
                            int8_t *inp2,
                            int8_t *inp3,
                            int8_t *inp4,
                            int8_t coefficients[], int nox, int rgb);

void subsample_y(int8_t *outp,
                 int8_t *inp0,
                 int8_t *inp1,
                 int8_t *inp2,
                 int8_t *inp3,
                 int8_t *inp4,
                 int8_t coefficients[], int nox) {
    for(int rgb = 0; rgb < 3; rgb++) {
        subsample_y_asm(outp,
                        inp0,
                        inp1,
                        inp2,
                        inp3,
                        inp4,
                        coefficients, nox/16, rgb);
        inp0 += SUBSAMPLE_MAX_OUTPUT_WIDTH;
        inp1 += SUBSAMPLE_MAX_OUTPUT_WIDTH;
        inp2 += SUBSAMPLE_MAX_OUTPUT_WIDTH;
        inp3 += SUBSAMPLE_MAX_OUTPUT_WIDTH;
        inp4 += SUBSAMPLE_MAX_OUTPUT_WIDTH;
    }
}

uint8_t gaussian[65] = {
  0,  0,  0,  0,  1,  1,  1,  2,  3,  4,  6,  8, 11, 15, 20, 27,
  35, 44, 55, 68, 83, 99,117,136,155,175,193,211,226,239,248,254,
  255,
  254,248,239,226,211,193,175,155,136,117, 99, 83, 68, 55, 44, 35,
  27, 20, 15, 11,  8,  6,  4,  3,  2,  1,  1,  1,  0,  0,  0,  0,
};


int round_down(int multiplier) {
    int v = (multiplier + 256) >> 9;
    if (v >  127) return  127;
    if (v < -128) return -128;
    return v;
}

static void calculate_ratios(int &ratio, int &ratio_inverse, int in_points, int out_points) {
//    printf("Points to point: %d to %d\n", in_points - 1, out_points - 1);
    ratio = 65536 * (in_points - 1) / (out_points - 1); // in Q.16 format
    ratio_inverse = 65536 * (out_points - 1) / (in_points - 1); // in Q.16 format
}

static int mkgaussian(int window_val[], int i, int start_x, int &pos_centre, int &int_pos_centre, int ratio, int ratio_inverse) {
    int window_width = (ratio >> 16) + 1;
    if (window_width > SUBSAMPLE_MAX_WINDOW_SIZE / 2) {
        window_width = SUBSAMPLE_MAX_WINDOW_SIZE / 2;
    }
    int sum = 0;
    pos_centre = i * ratio + (start_x << 16);
    int_pos_centre = pos_centre & 0xFFFF0000;
    for(int window_index = - window_width; window_index <= window_width; window_index++) {
        int pos = pos_centre - (window_index << 16);
        int64_t l = ((int64_t)(pos - int_pos_centre)) * ratio_inverse*2;
        int location_in_window = (l + (1<<27)) >> 28;
        int gauss = 0;
        if (location_in_window >= -32 && location_in_window <= 32) {
            gauss = gaussian[location_in_window + 32];
        }
        window_val[window_index + window_width] = gauss;
        sum += gauss;
    }
    int normalisation = 0x2000000 / sum;
    for(int k = 0; k < 2 * window_width + 1; k++) {
        window_val[k] = (window_val[k] * normalisation) >> 16;
    }
    return window_width;
}

void build_x_coefficients_strides(int8_t x_coefficients[32*SUBSAMPLE_MAX_OUTPUT_WIDTH*3],
                                  uint32_t strides[SUBSAMPLE_MAX_OUTPUT_WIDTH],
                                  int start_x, int end_x, int points) {
    int ratio, ratio_inverse;
    int window_val[SUBSAMPLE_MAX_WINDOW_SIZE];

    memset(x_coefficients, 0, 32*SUBSAMPLE_MAX_OUTPUT_WIDTH*3);
    calculate_ratios(ratio, ratio_inverse, end_x - start_x, points);

    int Y[3] = {64,  64,  64};   // Equal contributions of Y to R, G, and B
    int U[3] = { 0, -25, 127};   // R gets no contribution from U, G and B do
    int V[3] = {73, -37,   0};   // B gets no contribution from V, R and G do
    for(int i = 0; i < points; i++) {
        int pos_centre, int_pos_centre;
        int window_width = mkgaussian(window_val, i, start_x, pos_centre, int_pos_centre, ratio, ratio_inverse);
        int left_point = (int_pos_centre >> 16) - window_width;
        int stride_point = left_point;
        if (stride_point < 0) {
            stride_point = 0;
        }
        int index = (i & ~0xF) + 15 - (i & 0xF);
        strides[index] = (stride_point >> 1) * 4;
        int registered_stride_point = (strides[index] / 4)*2;
        int print = 0 && (strides[index] == 0);
        if (print) printf("pos_centre %f %d\n", pos_centre / 65536.0, int_pos_centre >> 16);
        if (print) printf("@@@@@@@@@@ %d %d\n", int_pos_centre >> 16, left_point);
        for(int rgb = 0; rgb < 3; rgb++) {
            int cindex = (i & ~0xF) *3 + 15 - (i & 0xF);
            for(int window_index = - window_width; window_index <= window_width; window_index++) {
                int gauss = window_val[window_index + window_width];
                if (print) printf("%d %d\n", window_index, gauss);
                int base_location = 2 * (left_point - registered_stride_point + window_width + window_index);
                if (base_location < 0) {
                    base_location = 0;
                } else if (base_location >= 32) {
                    base_location = 30;
                }
                int coefficient_location = base_location;
                int coefficient_base =  (cindex + rgb*16)*32;
                int coefficient_location_plus_one = coefficient_location + 1;
                int coefficient_location_minus_one = coefficient_location - 1;
                if (coefficient_location_minus_one < 0) {
                    coefficient_location_minus_one += 4;
                }
                coefficient_location_minus_one += coefficient_base;
                coefficient_location           += coefficient_base;
                coefficient_location_plus_one  += coefficient_base;
                if (print) {
                    printf("** %d %d %d", coefficient_location_minus_one, coefficient_location, coefficient_location_plus_one);
                    printf("   %d %d\n", rgb, window_index);
                }
                x_coefficients[coefficient_location] += round_down(gauss * Y[rgb]);
                int Vval = round_down(gauss * V[rgb]);
                int Uval = round_down(gauss * U[rgb]);
                
                if (base_location & 2) { // blue may overflow to 127+
                    x_coefficients[coefficient_location_plus_one] += Uval;
                    if (x_coefficients[coefficient_location_plus_one] < 0 && rgb == 2) {
                        x_coefficients[coefficient_location_plus_one] = 127;
                    }
                    x_coefficients[coefficient_location_minus_one] += Vval;
                } else {
                    x_coefficients[coefficient_location_plus_one] += Vval;
                    x_coefficients[coefficient_location_minus_one] += Uval;
                    if (x_coefficients[coefficient_location_minus_one] < 0 && rgb == 2) {
                        x_coefficients[coefficient_location_minus_one] = 127;
                    }
                }
            }
        }
    }
}


#define SUBSAMPLE_MAX_WINDOW_SIZE  5

void build_y_coefficients_strides(int8_t y_coefficients[16*SUBSAMPLE_MAX_OUTPUT_HEIGHT*SUBSAMPLE_MAX_WINDOW_SIZE],
                                  uint32_t strides[SUBSAMPLE_MAX_OUTPUT_HEIGHT],
                                  int start_y, int end_y, int points) {
    int window_val[SUBSAMPLE_MAX_WINDOW_SIZE];
    int ratio, ratio_inverse;
    memset(y_coefficients, 0, 16*SUBSAMPLE_MAX_OUTPUT_HEIGHT*SUBSAMPLE_MAX_WINDOW_SIZE);
    calculate_ratios(ratio, ratio_inverse, end_y - start_y, points);
//    printf("Ratio %08x\n", ratio);
    for(int index = 0; index < points; index++) {
        int pos_centre, int_pos_centre;
        int window_width = mkgaussian(window_val, index, start_y, pos_centre, int_pos_centre, ratio, ratio_inverse);
        int top_point = (int_pos_centre >> 16) - window_width;
        int stride_point = top_point;
        if (stride_point < 0) {
            stride_point = 0;
        }
        strides[index] = stride_point + window_width * 2;
//        printf("Stride %d: %d\n", index, stride_point);
        for(int window_index = - window_width; window_index <= window_width; window_index++) {
            int gauss = window_val[window_index + window_width];
            int base_location = top_point - stride_point + index;
            if (base_location < 0) {
                base_location = 0;
            }
            for(int l = 0; l < 16; l++) {
                int ci = (base_location * SUBSAMPLE_MAX_WINDOW_SIZE + window_index + window_width)*16 + l;
                y_coefficients[ci] += (gauss + 4) >> 3;
                if (y_coefficients[ci] < 0) {
                    y_coefficients[ci] = 127;
                }
            }
        }
    }
}

#if MAIN_TEST

uint8_t inputs[38400] = {
    #include "yuv.h"
};

uint8_t morph(uint8_t x) {
    int z = x;
    z = z - 128;
    if (z < 0) {
        z += 256;
    }
    return z;
}

int main(void) {
    uint8_t line[320];
    int8_t outp[SUBSAMPLE_MAX_OUTPUT_WIDTH][3];
    int8_t outp3[SUBSAMPLE_MAX_OUTPUT_WIDTH][3];
    int8_t outp2[48][3][SUBSAMPLE_MAX_OUTPUT_WIDTH];
    int8_t x_coefficients[32*SUBSAMPLE_MAX_OUTPUT_WIDTH*3];
    int8_t y_coefficients[16*SUBSAMPLE_MAX_OUTPUT_HEIGHT*SUBSAMPLE_MAX_WINDOW_SIZE];
    uint32_t x_strides[SUBSAMPLE_MAX_OUTPUT_WIDTH];
    uint32_t y_strides[SUBSAMPLE_MAX_OUTPUT_HEIGHT];

            asm volatile("ldc r11, 0x200; vsetc r11");

    for(int y = 0; y < 120; y++) {
        for(int i = 0; i < SUBSAMPLE_MAX_OUTPUT_WIDTH; i+=2) {
            int index = (y * SUBSAMPLE_MAX_OUTPUT_WIDTH + i)*2;
            inputs[index] = i == y ? 128 : 127;
            inputs[index+1] = -60+y;
            inputs[index+2] = i+1 == y ? 128 : 127;
            inputs[index+3] = -80+i;
        }
    }
    if(0)for(int yyy = 38; yyy<44; yyy++) {
        build_y_coefficients_strides(y_coefficients, y_strides, 7, yyy, 32);
        for(int i = 0; i < 32; i++) {
            for(int window = 0; window < SUBSAMPLE_MAX_WINDOW_SIZE; window++) {
                printf(" %4d", y_coefficients[(i*5 + window)*16 + 0]);
            }
            printf("\n");
        }
        printf("\n");
    }
    
#define YYY 39
    build_y_coefficients_strides(y_coefficients, y_strides, 7, YYY, 32);
    if(0)for(int i = 0; i < 16; i++) {
        printf("**%d\n", y_strides[i]);
        for(int window = 0; window < SUBSAMPLE_MAX_WINDOW_SIZE; window++) {
            for(int j = 0; j < 16; j++) {
                printf(" %4d", y_coefficients[(i*5 + window)*16 + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
//    build_x_coefficients_strides(x_coefficients, x_strides, 3, 40, 32);
    if(0)for(int i = 13; i < 16; i++) {
        printf("**%d\n", x_strides[i]);
        for(int j = 0; j < 16; j++) {
            for(int rgb = 0; rgb < 3; rgb++) {
                int index = (i & ~0xF)*3 + (i & 0xF);
                printf(" %4d", x_coefficients[(index + rgb*16)*32 + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("P3\n32 32 255\n");
    for(int j = 0; j < 48; j++) {
        for(int i = 0; i < 32; i++) {
            outp2[j][0][i] = i == 16 || j == 38 || j == 32 ? 127 : -127;
            outp2[j][1][i] = (j & 2) ? 127 : -127;
            outp2[j][2][i] = (j & 4) ? 127 : -127;
        }
    }
//        for(int i = 0; i < 320; i++) {
//            line[i] = morph(inputs[index]);
//            index++;
//        }
//        transform_line(outp, line);
//        subsample_x(outp2[j], line, x_coefficients, x_strides, 32);
    memset(outp, 0, sizeof(outp));
    int ocnt =  0;
    int yindex = 0;
    for(int j = 0; j <= YYY+5; j++) {
        while(j == y_strides[ocnt]) {
            transform_line_y(outp,
                             outp2[j >= 4 ? j-4 : 0],
                             outp2[j >= 3 ? j-3 : 0],
                             outp2[j >= 2 ? j-2 : 0],
                             outp2[j >= 1 ? j-1 : 0],
                             outp2[j-0],
                             &y_coefficients[yindex], 32);
            subsample_y(outp3,
                        outp2[j >= 4 ? j-4 : 0],
                        outp2[j >= 3 ? j-3 : 0],
                        outp2[j >= 2 ? j-2 : 0],
                        outp2[j >= 1 ? j-1 : 0],
                        outp2[j-0],
                        &y_coefficients[yindex], 32);
            for(int i = 0; i < 32; i++) {
                for(int j = 0; j < 3; j++) {
                    if (outp3[i][j] != outp[i][j]) {
                        printf("Diff %d %d   %d %d\n", i, j, outp3[i][j], outp[i][j]);
                    }
                }
            }
            for(int i = 0; i < 32; i++) {
//            printf("\n& %08x\n", &outp[i][0]);
                printf("%d %d %d ", outp[i][0]+128,  outp[i][1]+128,  outp[i][2]+128);
            }
            printf("\n");
            ocnt++;
            yindex += 16*5;
        }
    }
    return 0;
}
#endif
