Python interface for XTFLM
=========================

Build
-----

To build use the following command sequence::

  (mkdir build && cd build && cmake .. && make install)

Usage
-----

Use it for example as follows::

  #!/usr/bin/env python
  # Copyright (c) 2020, XMOS Ltd, All rights reserved

  import sys
  import numpy as np
  import cv2
  from xtflm_interpreter import XTFLMInterpreter

  ie = XTFLMInterpreter(model_path = sys.argv[1])

  img = cv2.imread(sys.argv[2])
  img = cv2.resize(img, (128,128))
  img = img[:, :, ::-1]  # Channel swapping due to mismatch between open CV and XMOS
  img = np.asarray(img, dtype=np.float32)
  img = img / 256.0

  ie.set_input_tensor(0, img)
  ie.invoke()
  answer = ie.get_output_tensor(0, tensor = np.zeros((10), dtype=np.float32))
  print(answer) 

Input and output can either be raw data or numpy arrays. The
get_output_tensor has an optional tensor argument that will cause it to
return a numpy array; otherwise it will return bytes().
