import json
import numpy as np


#
tfl_to_xcore_map = {1:3, 2:6, 3:9, 4:12, 6:15, 7:18, 8:21, 9:24, 10:27, 11:28}


# Opening JSON file
f = open('tflite.json')
tflite_data = json.load(f)
# Closing file
f.close()

f = open('xcore.json')
xcore_data = json.load(f)
# Closing file
f.close()

# Iterating through the json
# list
for i in tfl_to_xcore_map:
    tfl = np.array(tflite_data[i]["data"][0]["val"])
    xc = np.array(xcore_data[tfl_to_xcore_map[i]]["data"][0]["val"])
    diffs = tfl - xc
    unique, counts = np.unique(diffs, return_counts=True)
    print("\n\nTFLite %s, node %d" %(tflite_data[i]["op"], i))
    print("Xcore %s, node %d" %(xcore_data[tfl_to_xcore_map[i]]["op"], tfl_to_xcore_map[i]))
    print(np.asarray((unique, counts)).T)
