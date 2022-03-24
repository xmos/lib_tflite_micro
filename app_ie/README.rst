app_ie: test program
====================

Env/Dependencies etc
--------------------

The short instructions::

       git clone  git@github.com:xmos/lib_tflite_micro.git
       cd lib_tflite_micro
       python3 fetch_dependencies.py
       git submodule update --init --recurse
       cd app_ie
       xmake
       xsim -t bin/app_inference_engine.xe


