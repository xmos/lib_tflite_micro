from xmos_ai_tools import xformer as xf

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

import os

input_shape = (160, 160, 3)
alpha_value = 1.0
model_weights = None
classes = 16
classifier_activation = None

# Create MobileNet model
model = MobileNetV2(
    input_shape=input_shape,
    alpha=alpha_value,
    weights=model_weights,
    classes=classes,
    classifier_activation=classifier_activation,
)

# Use tf lite onverter to quantize the model to int8
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the optimization mode
converter.optimizations = [tf.lite.Optimize.DEFAULT]


# Use a representative dataset generator for proper INT8 optimization
def representative_dataset_gen():
    for _ in range(100):
        yield [tf.random.normal([1] + list(input_shape))]


converter.representative_dataset = representative_dataset_gen

# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8  #

# quantize the model
tflite_model = converter.convert()

# Save the quantized model
with open("mobilenetV2_input_160x160x3_alpha_1.tflite", "wb") as f:
    f.write(tflite_model)

print("Model quantized and saved successfully.")

# split the model
xf.convert(
    "mobilenetV2_input_160x160x3_alpha_1.tflite",
    "split_mobilenetV2_input_160x160x3_alpha_1.tflite",
    {
        "xcore-op-split": "",
        "xcore-op-split-top-op": "0,7,11",
        "xcore-op-split-bottom-op": "6,9,14",
        "xcore-op-split-num-splits": "9,3,3",
    },
)

opt_params_path = "1.params"
naming_prefix = "model_"

# optimize the model
xf.convert(
    "split_mobilenetV2_input_160x160x3_alpha_1.tflite",
    "src/xcore_optimised_mobilenetV2_input_160x160x3_alpha_1.tflite",
    {
        "xcore-offline-offsets": "",
        "xcore-flash-image-file": opt_params_path,
    },
)

xf.generate_flash(
    output_file="xcore_flash_binary.out",
    model_files=["src/xcore_optimised_mobilenetV2_input_160x160x3_alpha_1.tflite"],
    param_files=["1.params"],
)

# clean up
# list of files to be deleted
files = [
    "split_mobilenetV2_input_160x160x3_alpha_1.tflite",
    "split_mobilenetV2_input_160x160x3_alpha_1.tflite.cpp",
    "split_mobilenetV2_input_160x160x3_alpha_1.tflite.h",
]

for file_path in files:
    try:
        os.remove(file_path)
    except Exception as e:
        print(f"An error occurred while removing {file_path}: {e}")
