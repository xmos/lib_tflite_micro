import numpy as np
import argparse
import tensorflow as tf


def load_raw_data(filename, dtype=np.int8):
    return np.fromfile(filename, dtype=dtype)


def save_raw_data(filename, data, dtype=np.int8):
    data.astype(dtype).tofile(filename)


def main(args):
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = load_raw_data(args.input_file).reshape(input_details[0]["shape"])
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    save_raw_data(args.output_file, output_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Feed raw input to a TFLite model and save the output."
    )
    parser.add_argument("model_path", type=str, help="Path to the .tflite model file.")
    parser.add_argument("input_file", type=str, help="Path to the raw input file.")
    parser.add_argument(
        "output_file", type=str, help="Path to save the raw output file."
    )
    args = parser.parse_args()
    main(args)
