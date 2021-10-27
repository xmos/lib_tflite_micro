#!/usr/bin/env python
# Copyright (c) 2020, XMOS Ltd, All rights reserved

import sys
import numpy as np
import cv2
from tflm_interpreter import TFLMInterpreter

ie = TFLMInterpreter(model_path = sys.argv[1])

img = cv2.imread(sys.argv[2])
img = cv2.resize(img, (128,128))
img = img[:, :, ::-1]  # Channel swapping due to mismatch between open CV and XMOS
img = np.asarray(img, dtype=np.float32)
img = img / 256.0

ie.set_input_tensor(0, img)
ie.invoke()
answer = ie.get_output_tensor(0)
print(answer)
