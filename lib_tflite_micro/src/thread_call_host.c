#ifndef __xcore__

#include "lib_tflite_micro/api/inference_engine.h"
#include "thread_call.h"

#include <stdint.h>
#include <stdio.h>
#include <assert.h>

int thread_invoke_5(struct inference_engine *ie, thread_info_t*ptr) {
    ptr->thread_ids.id[0] = 0;
    ptr->thread_ids.id[1] = 1;
    ptr->thread_ids.id[2] = 2;
    ptr->thread_ids.id[3] = 3;
    return interp_invoke_internal(ie);
}

int thread_invoke_4(struct inference_engine *ie, thread_info_t*ptr) {
    ptr->thread_ids.id[0] = 0;
    ptr->thread_ids.id[1] = 1;
    ptr->thread_ids.id[2] = 2;
    ptr->thread_ids.id[3] = -1;
    return interp_invoke_internal(ie);
}

int thread_invoke_3(struct inference_engine *ie, thread_info_t*ptr) {
    ptr->thread_ids.id[0] = 0;
    ptr->thread_ids.id[1] = 1;
    ptr->thread_ids.id[2] = -1;
    ptr->thread_ids.id[3] = -1;
    return interp_invoke_internal(ie);
}

int thread_invoke_2(struct inference_engine *ie, thread_info_t*ptr) {
    ptr->thread_ids.id[0] = 0;
    ptr->thread_ids.id[1] = -1;
    ptr->thread_ids.id[2] = -1;
    ptr->thread_ids.id[3] = -1;
    return interp_invoke_internal(ie);
}

int thread_invoke_1(struct inference_engine *ie, thread_info_t*ptr) {
    ptr->thread_ids.id[0] = -1;
    ptr->thread_ids.id[1] = -1;
    ptr->thread_ids.id[2] = -1;
    ptr->thread_ids.id[3] = -1;
    return interp_invoke_internal(ie);
}

#endif
