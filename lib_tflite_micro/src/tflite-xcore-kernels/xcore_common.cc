#ifndef __xcore__

#include "../thread_call.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

extern "C" void DebugLog(const char *s) {
  while (*s) {
    putchar(*s);
    s++;
  }
} // Not sure why we need this

//////

void thread_init_1(thread_info_t *ptr) {}
void thread_init_2(thread_info_t *ptr) {}
void thread_init_3(thread_info_t *ptr) {}
void thread_init_4(thread_info_t *ptr) {}
void thread_init_5(thread_info_t *ptr) {}
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