Command line interface for XTFLM
===============================

Build
-----

From the repository root, configure and build the interpreter::

  cmake -S host_cmd_line_interpreter -B host_cmd_line_interpreter/build
  cmake --build host_cmd_line_interpreter/build --parallel 8

To install the executable into ``host_cmd_line_interpreter/bin``::

  cmake --build host_cmd_line_interpreter/build --target install --parallel 8

Run the MobileNet test after building::

  ctest --test-dir host_cmd_line_interpreter/build --output-on-failure

Usage
-----

Use it in either of the two following ways::

  host_cmd_line_interpreter/bin/xtflm_interpreter_cmdline model.tflite input-file output-file
  host_cmd_line_interpreter/bin/xtflm_interpreter_cmdline model.tflite -i files ... -o files

input and output are raw data. The first form only works when the network
expects a single input and has a single output. The second form works with
any number of inputs and outputs
