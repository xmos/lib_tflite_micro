#include <cstdio>
#include <cstring>
#include <algorithm>

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

namespace tflite {
namespace ops {
namespace micro {
namespace xcore {

void calculateThreadSplit(int tc, int split_size, int split_start[],
                          int split_end[]) {
  split_start[0] = 0;

  // Alignment is to four
  // Figure out min number of threads needed while keeping alignment
  // By dividing split_size by four and ceil that
  tc = std::min(tc, (split_size + 3) >> 2);

  for (int i = 0; i < tc; i++) {
    auto split = (split_size + (tc - i) - 1) / (tc - i);
    split_size -= split;
    if (split > 0) {
      split_end[i] = split_start[i] + split;
      if (i != tc - 1)
        split_start[i + 1] = split_end[i];
    } else {
      break;
    }
  }

  // Align up or down split_starts to word length = 4 bytes,
  // so that each thread begins work at an aligned address
  // The last thread handles remaining items, so don't modify the end
  for(int i = 1; i < tc; i++) {
    if((split_start[i] & 3) >= 3) {
      // Align up
      split_start[i] = (split_start[i] + 3) & ~3;
    } else {
      // Align down
      split_start[i] = split_start[i] & ~3;
    }
    split_end[i-1] = split_start[i];
  }
}

} // namespace xcore
} // namespace micro
} // namespace ops
} // namespace tflite

#define MAX_DEBUG_LOG_LENGTH 256
#define MAX_DEBUG_LOG_ENTRIES 3

int debug_log_index = 0;
char ALIGN(4) debug_log_buffer[MAX_DEBUG_LOG_LENGTH * MAX_DEBUG_LOG_ENTRIES];

extern "C" void DebugLog(const char *s) {
  strcpy(&debug_log_buffer[debug_log_index * MAX_DEBUG_LOG_ENTRIES], s);
  printf("%s", &debug_log_buffer[debug_log_index * MAX_DEBUG_LOG_ENTRIES]);
  debug_log_index++;
  if (debug_log_index == MAX_DEBUG_LOG_ENTRIES)
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
