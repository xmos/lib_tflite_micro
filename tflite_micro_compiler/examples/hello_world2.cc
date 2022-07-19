/* Copyright 2020 Christof Petig. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iostream>

#include "compiled_hello_world.cc.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

void test_compiled(void) {
  hello_init();
  tflite::GetTensorData<float>(hello_input(0))[0] = 1.57f;
  hello_invoke();
  float out = tflite::GetTensorData<float>(hello_output(0))[0];
  std::cerr << "result " << out << std::endl;
}

int main(int argc, char** argv) {
  test_compiled();
  return 0;
}
