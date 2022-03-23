#!/usr/bin/env python
# Copyright (c) 2020, XMOS Ltd, All rights reserved
import sys
import numpy as np
import cv2
from xtflm_interpreter import XTFLMInterpreter

from xmos_ai_tools import xformer as xf

xf.convert("./mobilenet_v1_0.25_128.tflite", "./xcore.tflite", params=None)
ie = XTFLMInterpreter()
ie.set_model("./xcore.tflite")

img = cv2.imread(sys.argv[1])
img = cv2.resize(img, (224,224))
# Channel swapping due to mismatch between open CV and XMOS
img = img[:, :, ::-1]  
img = np.asarray(img-128, dtype=np.int8)
# img = img / 256

ie.set_input_tensor(0, img)

ie.invoke()

answer = ie.get_output_tensor(0, tensor = np.zeros((10), dtype=np.float32))
print("\n", answer)

ie.close()