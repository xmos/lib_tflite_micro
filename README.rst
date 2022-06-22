TFLite-micro container
======================

This repo is a wrapper for lib_tflite_micro.
This repo contains all third party repos that are needed to use tflite-micro on an XCORE.
It wraps this third-party C++ software up in a library that exposes three interfaces:

#. A C interface for use on embedded systems (``lib_tflite_micro``).

#. A command line interface for use on a host (``host_command_line_interface``)

The ``lib_tflite_micro`` library depends on ``lib_nn``.

It provides the following services:

* lib_tflite_micro/module_build_info: file that allows lib_tflite_micro to be integrated into normal XMOS build flow

* lib_tflite_micro/src: a function that wraps the C++ interpreter in C (inference_engine.cc), and a collection of
  kernels that we add to tflite-micro with XCORE specific operators
  
* lib_tflite_micro/api: .h files for the above

* host_cmd_line_interpeter: a command line wrapper for XTFLM, enabling it to be used over the command line.


C interface
-----------

It exposes a C interface comprising a datastructure (inference_engine_t)
with a few functions that can be used to initialise the structure and/or


Getting the XCORE.AI optimiser
------------------------------

You can get the XCORE.AI optimiser through pypi:

* https://pypi.org/project/xmos-ai-tools/

This gets you both a command line interface and python interface to the xcore-opt tool that optimises
a ``.tflite`` file for xcore
perform an inference. The data structure itself can be used to directly
read/write data into tensors, this enables sensors to directly operate
in the tensor space.

The C interface can be used with the standard XMOS build system, and is
built from the appropriate application directory

Command line interface
----------------------

The command line interface uses the C interface above and makes it accessible
from the command line, enabling the end user to send data through a TFLite model
using the XTFLM interpreter. The XTFLM intepreter will have XCORE specific operators
(such as 2D convolutions, loading from flash) that are emulated on the host.

The command line interface is built by invoking ``make install`` at top level or
inside ``host_command_line_interface``.

The command line interface cane be tested by invoking ``make test`` at either level. 
