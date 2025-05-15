#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <algorithm>

#if defined __GNUC__
#define ALIGN(X) __attribute__((aligned(X)))
#elif defined _MSC_VER
#define ALIGN(X) __declspec(align(X))
#elif defined __TASKING__
#define ALIGN(X) __align(X)
#endif

#define MAX_DEBUG_LOG_LENGTH 256
#define MAX_DEBUG_LOG_ENTRIES 3

int debug_log_index = 0;
char ALIGN(4) debug_log_buffer[MAX_DEBUG_LOG_LENGTH * MAX_DEBUG_LOG_ENTRIES];

extern "C" void DebugLog(const char* format, va_list args) {
  vsnprintf(&debug_log_buffer[debug_log_index * MAX_DEBUG_LOG_ENTRIES], MAX_DEBUG_LOG_LENGTH, format, args);
  printf("%s", &debug_log_buffer[debug_log_index * MAX_DEBUG_LOG_ENTRIES]);
  debug_log_index++;
  if (debug_log_index == MAX_DEBUG_LOG_ENTRIES)
    debug_log_index = 0;
}

