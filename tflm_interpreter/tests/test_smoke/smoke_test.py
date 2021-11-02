#!/usr/bin/env python
# Copyright (c) 2020, XMOS Ltd, All rights reserved

import sys
import numpy as np
import cv2
from tflm_interpreter import TFLMInterpreter

ie = TFLMInterpreter(model_path = sys.argv[1], params_path = sys.argv[2])

with open(sys.argv[3], 'rb') as fd:
    img = fd.read()

ie.set_input_tensor(0, img)
ie.invoke()
answer1 = ie.get_output_tensor(0)
answer2 = ie.get_output_tensor(1)
with open(sys.argv[4], 'wb') as fd:
    fd.write(answer1)
with open(sys.argv[5], 'wb') as fd:
    fd.write(answer2)
