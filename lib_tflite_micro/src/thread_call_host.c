#ifndef __xcore__

#include "inference_engine.h"
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

static void * args[4][10];
static int32_t max_thread_id = -1;
void thread_variable_setup(void *arg1, void *arg2, uint32_t thread_id) {
    assert(thread_id != -1);
    args[thread_id][1] = arg1;
    args[thread_id][2] = arg2;
    if ((int)thread_id > max_thread_id) {
        max_thread_id = thread_id;
    }
}

void thread_call(void *arg0, void *arg1, void *arg2,
                 thread_function_pointer_t fp, thread_info_t *ptr) {
    (*fp)(arg0, arg1, arg2);
    for(int i = 0; i <= max_thread_id; i++) {
        (*fp)(arg0, args[i][1], args[i][2]);
    }
    max_thread_id = -1;
}

#endif
