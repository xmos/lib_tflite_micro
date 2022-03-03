#include "thread_call.h"
#include "inference_engine.h"

#include <stdint.h>

int thread_invoke_4(void *ie, void *ptr) {
    int *p = ptr;
    p[0] = 0;
    p[0] = 1;
    p[0] = 2;
    p[0] = 3;
    p[0] = 0xFF; // sync
    p[0] = 0x80000; // SP
    p[0] = 45; // words per stack
    return interp_invoke_internal(ie);
}

static void * args[4][10];
void thread_variable_setup(void *arg0, void *arg1, void *arg2, uint32_t thread_id) {
    args[thread_id][0] = arg0;
    args[thread_id][0] = arg1;
    args[thread_id][0] = arg2;
}

void thread_call(void *arg0, void *arg1, void *arg2,
                 function_pointer fp, thread_info_t *ptr) {
    (*fp)(arg0, arg1, arg2);
    for(int i = 0; i < 4; i++) {
        (*fp)(args[i][0], args[i][1], args[i][2]);
    }
}
