lib_tflite_micro
================


This repo is a wrapper for lib_tflite_micro. It provides the following services:

* lib_tflite_micro/module_build_info: file that allows lib_tflite_micro to be integrated into normal XMOS build flow

* lib_tflite_micro/src: a function that wraps the C++ interpreter in C (inference_engine.cc), and a collection of
  kernels that we add to tflite-micro with XCORE specific operators
  
* lib_tflite_micro/api: .h files for the above

* tflm_interpreters: a wrapper of the full interpreter for Python, enabling TFLM to be instantiated from, for example pytest.

* host_cmd_line_interpeter: a command line wrapper for TFLM, enabling it to be used over the command line.
