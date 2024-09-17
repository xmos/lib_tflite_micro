#include "xcore_error_reporter.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

#ifdef __xcore__
#include<print.h>
#endif

//#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
#include "tensorflow/lite/micro/debug_log.h"
//#endif

namespace tflite_micro {
namespace micro {
namespace xcore {

void XCoreErrorReporter::Init(char *debugBuffer, int debugBufferLength) {
  buffer = debugBuffer;
  max_len = debugBufferLength;
  memset(debugBuffer, 0, max_len);
}

void XCoreErrorReporter::Log(const char *format, va_list args) {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
  static constexpr int kMaxLogLen = 256;
  if (len + kMaxLogLen > max_len) {
    int new_len = max_len - kMaxLogLen;
    if (new_len < 0) {
      new_len = 0;
    }
    for (int i = 0; i <= new_len; i++) {
      buffer[i] = buffer[i - new_len + len];
    }
    len = new_len;
  }
  vsprintf(buffer + len, format, args);
  len = strlen(buffer);
#ifdef __xcore__
  printstr(buffer);
#else
  printf("%s", buffer);
#endif

#endif
}

int XCoreErrorReporter::Report(const char *format, va_list args) {
  Log(format, args);
  return 0;
}

} // namespace xcore
} // namespace micro
} // namespace tflite_micro
