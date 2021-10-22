Command line interface for TFLM
===============================

Build
-----


To build use the following command sequence::

  (mkdir build && cd build && cmake .. && make install)

Usage
-----

Use it in either of the two following ways::

  bin/tflm_interpreter_cmdline  model.tflite input-file output-file
  bin/tflm_interpreter_cmdline  model.tflite -i files ... -o files 

input and output are raw data. The first form only works when the network
expects a single input and has a single output. The second form works with
any number of inputs and outputs
