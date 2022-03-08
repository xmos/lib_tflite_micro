Command line interface for XTFLM
===============================

Build
-----


To build use the following command sequence::

  (mkdir build && cd build && cmake .. && make install)

Usage
-----

Use it in either of the two following ways::

  bin/xtflm_interpreter_cmdline  model.tflite input-file output-file
  bin/xtflm_interpreter_cmdline  model.tflite -i files ... -o files 

input and output are raw data. The first form only works when the network
expects a single input and has a single output. The second form works with
any number of inputs and outputs
