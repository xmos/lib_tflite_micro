// Copyright (c) 2019, XMOS Ltd, All rights reserved

#ifndef XCORE_ERROR_REPORTER_H_
#define XCORE_ERROR_REPORTER_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"

namespace tflite {
namespace micro {
namespace xcore {

class XCoreErrorReporter : public tflite::MicroErrorReporter {
public:
  explicit XCoreErrorReporter(){};
  ~XCoreErrorReporter() override = default;
  void Init(char *debugBuffer, int debugBufferLength);
  void Log(const char *format, va_list args);
  int Report(const char *format, va_list args) override;

private:
  char *buffer;
  int max_len;
  int len = 0;
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

} // namespace xcore
} // namespace micro
} // namespace tflite

#endif // XCORE_ERROR_REPORTER_H_
