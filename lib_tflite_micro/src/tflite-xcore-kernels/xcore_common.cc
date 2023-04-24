#include <cstdio>
#include <cstring>

#define MAX_DEBUG_LOG_LENGTH  256
#define MAX_DEBUG_LOG_ENTRIES 3

int debug_log_index = 0;
char debug_log_buffer[MAX_DEBUG_LOG_LENGTH * MAX_DEBUG_LOG_ENTRIES] __attribute__((aligned(4)));

extern "C" void DebugLog(const char* s)
{
    strcpy(&debug_log_buffer[debug_log_index * MAX_DEBUG_LOG_ENTRIES], s);
    printf("%s", &debug_log_buffer[debug_log_index * MAX_DEBUG_LOG_ENTRIES]);
    debug_log_index++;
    if(debug_log_index == MAX_DEBUG_LOG_ENTRIES)
        debug_log_index = 0;
}

#ifndef __xcore__

#include "../thread_call.h"
#include <assert.h>

//////

void thread_init_1(thread_info_t *ptr) {
    ptr->thread_ids.id[0] = -1;
    ptr->thread_ids.id[1] = -1;
    ptr->thread_ids.id[2] = -1;
    ptr->thread_ids.id[3] = -1;
}

void thread_init_2(thread_info_t *ptr) {
    ptr->thread_ids.id[0] = 0;
    ptr->thread_ids.id[1] = -1;
    ptr->thread_ids.id[2] = -1;
    ptr->thread_ids.id[3] = -1;
}

void thread_init_3(thread_info_t *ptr) {
    ptr->thread_ids.id[0] = 0;
    ptr->thread_ids.id[1] = 1;
    ptr->thread_ids.id[2] = -1;
    ptr->thread_ids.id[3] = -1;
}
void thread_init_4(thread_info_t *ptr) {
    ptr->thread_ids.id[0] = 0;
    ptr->thread_ids.id[1] = 1;
    ptr->thread_ids.id[2] = 2;
    ptr->thread_ids.id[3] = -1;
}

void thread_init_5(thread_info_t *ptr) {
    ptr->thread_ids.id[0] = 0;
    ptr->thread_ids.id[1] = 1;
    ptr->thread_ids.id[2] = 2;
    ptr->thread_ids.id[3] = 3;
}

void thread_destroy(thread_info_t *ptr) {}

static void *args[4][10];
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
  for (int i = 0; i <= max_thread_id; i++) {
    (*fp)(arg0, args[i][1], args[i][2]);
  }
  max_thread_id = -1;
}

#endif