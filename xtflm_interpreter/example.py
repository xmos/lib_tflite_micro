#!/usr/bin/env python
# Copyright (c) 2020, XMOS Ltd, All rights reserved
import sys
import numpy as np
import cv2
from xtflm_interpreter import XTFLMInterpreter

from xmos_ai_tools import xformer as xf
xf.convert("./mobilenet_v1_0.5_224.tflite", "./xcore.tflite", params=None)
ie = XTFLMInterpreter()
ie.set_model("./xcore.tflite")
ie.initialise_interpreter(engine_num=0)
img = cv2.imread(sys.argv[1])
img = cv2.resize(img, (224,224))
img = img[:, :, ::-1]  # Channel swapping due to mismatch between open CV and XMOS
img = np.asarray(img-128, dtype=np.int8)
# img = img / 256
ie.set_input_tensor(0, img)
print(ie.get_input_tensor())

print("\n",ie.get_input_details())
print("\nInput Size: ", ie.get_input_tensor_size())
print("\nOutput Size: ", ie.get_output_tensor_size())
print("\nTensor Arena Size: ", ie.tensor_arena_size())
# print(ie.get_input_tensor())
# print(ie.get_input_size())
# print(ie.get_input_detais())
ie.invoke()
answer = ie.get_output_tensor(0, tensor = np.zeros((10), dtype=np.float32))
print("\n", answer)

print("\n", ie.get_output_details())

ie.close()