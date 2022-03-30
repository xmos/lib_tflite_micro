#!/usr/bin/env python
# Copyright (c) 2020, XMOS Ltd, All rights reserved

import sys, os
import numpy as np
import cv2

from xinterpreters import XCORE_TFLM_Host_Interpreter

ie = XCORE_TFLM_Host_Interpreter()
ie.set_model(model_path = "./xinterpreters/host/tests/test_smoke/smoke_model.tflite", params_path = "./xinterpreters/host/tests/test_smoke/smoke_model.flash")
with open("./xinterpreters/host/tests/test_smoke/detection_0.raw", 'rb') as fd:
    img = fd.read()

ie.set_input_tensor(0, img)
ie.invoke()

answer1 = ie.get_output_tensor(0)
answer2 = ie.get_output_tensor(1)
with open("./xinterpreters/host/tests/test_smoke/out0", 'wb') as fd:
    fd.write(answer1)
with open("./xinterpreters/host/tests/test_smoke/out1", 'wb') as fd:
    fd.write(answer2)