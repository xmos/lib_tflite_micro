#!/usr/bin/env python
# Copyright (c) 2020, XMOS Ltd, All rights reserved

import sys
import numpy as np
import cv2
from xtflm_interpreter import XTFLMInterpreter

ie = XTFLMInterpreter(model_path = sys.argv[1], params_path = sys.argv[2])

with open(sys.argv[3], 'rb') as fd:
    img = fd.read()

ie.set_input_tensor(0, img)
ie.invoke()
output_num = 0
for arg in sys.argv[4:]:
    answer = ie.get_output_tensor(output_num)
    with open(arg, 'wb') as fd:
        fd.write(answer)
    output_num += 1
