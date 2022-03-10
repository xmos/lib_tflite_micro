#ifndef __xcore__

#include "thread_call.h"
#include "inference_engine.h"

#include <stdint.h>
#include <stdio.h>

int thread_invoke_4(void *ie, void *ptr) {
    int *p = ptr;
    p[0] = 0;
    p[1] = 1;
    p[2] = 2;
    p[3] = 3;
    p[4] = 0xFF; // sync
    p[5] = 0x80000; // SP
    p[6] = 45; // words per stack
    return interp_invoke_internal(ie);
}

static void * args[4][10];
static int32_t max_thread_id = -1;
void thread_variable_setup(void *arg0, void *arg1, void *arg2, uint32_t thread_id) {
    args[thread_id][0] = arg0;
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
        (*fp)(args[i][0], args[i][1], args[i][2]);
    }
    max_thread_id = -1;
}
#endif
