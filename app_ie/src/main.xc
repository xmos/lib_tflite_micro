// Copyright (c) 2020, XMOS Ltd, All rights reserved

#include <platform.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xclib.h>
#include <stdint.h>
#include "inference_engine.h"
#include "server_memory.h"
#include "one_tflite.h"
#include "five_tflite.h"
#include "input_data.h"
#include "output_data.h"

int main(void) 
{
    par 
    {
        on tile[0]: {
            int error = 0;
            unsafe {
#define MODEL one_tflite
                inference_engine_t ie;
                inference_engine_initialize_with_memory(&ie);
                
                inference_engine_load_model(&ie, sizeof(MODEL), (MODEL, uint32_t[]), null);
                memcpy(ie.input_buffers[0], input_data, sizeof(input_data));
                int t0, t1;
                asm volatile ("gettime %0" : "=r" (t0));
                interp_invoke(&ie);
                asm volatile ("gettime %0" : "=r" (t1));
                printf("Inferred single threaded model on one thread   %d\n", t1-t0);
                for(int i = 0; i < sizeof(output_data); i++) {
                    if (output_data[i] != ((uint8_t *)ie.output_buffers[0])[i]) {
                        printf("Wrong data index %d byte %2x should be %02x\n", i, ((uint8_t *)ie.output_buffers[0])[i], output_data[i]);
                        error = 1;
                    }
                }
            }
            unsafe {
#define MODEL one_tflite
                inference_engine_t ie;
                inference_engine_initialize_with_memory(&ie);
                
                inference_engine_load_model(&ie, sizeof(MODEL), (MODEL, uint32_t[]), null);
                memcpy(ie.input_buffers[0], input_data, sizeof(input_data));
                int t0, t1;
                asm volatile ("gettime %0" : "=r" (t0));
                interp_invoke_par_5(&ie);
                asm volatile ("gettime %0" : "=r" (t1));
                printf("Inferred single threaded model on five threads %d\n", t1-t0);
                for(int i = 0; i < sizeof(output_data); i++) {
                    if (output_data[i] != ((uint8_t *)ie.output_buffers[0])[i]) {
                        printf("Wrong data index %d byte %2x should be %02x\n", i, ((uint8_t *)ie.output_buffers[0])[i], output_data[i]);
                        error = 1;
                    }
                }
            }
            unsafe {
#undef MODEL
#define MODEL five_tflite
                inference_engine_t ie;
                inference_engine_initialize_with_memory(&ie);
                
                inference_engine_load_model(&ie, sizeof(MODEL), (MODEL, uint32_t[]), null);
                memcpy(ie.input_buffers[0], input_data, sizeof(input_data));
                int t0, t1;
                asm volatile ("gettime %0" : "=r" (t0));
                interp_invoke_par_5(&ie);
                asm volatile ("gettime %0" : "=r" (t1));
                printf("Inferred five threaded model on five threads   %d\n", t1-t0);
                for(int i = 0; i < sizeof(output_data); i++) {
                    if (output_data[i] != ((uint8_t *)ie.output_buffers[0])[i]) {
                        printf("Wrong data index %d byte %2x should be %02x\n", i, ((uint8_t *)ie.output_buffers[0])[i], output_data[i]);
                        error = 1;
                    }
                }
            }
            if (error) {
                printf("ERRORs spotted\n");
            } else {
                printf("Done\n");
            }
        }
    }
    return 0;
}
