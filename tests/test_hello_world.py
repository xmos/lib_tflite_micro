from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from conftest import build_run_sim

CWD = Path(__file__).resolve().parent

INPUT_FILE = "input.bin"
OUTPUT_FILE = "output.bin"
PLOT_FILE = "comparison_{hw_target}.png"
MAX_MSE = 0.01
N_SINES = 2

# Quantization parameters of the checked-in generated model.
INPUT_SCALE = 0.024480115622282028
INPUT_ZERO_POINT = -128
OUTPUT_SCALE = 0.0082909567281603813
OUTPUT_ZERO_POINT = 5

def generate_input(build_dir: Path):
    angles = np.linspace(0, N_SINES * 2 * np.pi, 256*N_SINES, endpoint=False)
    expected = np.sin(angles)
    # The model accepts one period, so wrap angles before quantizing.
    wrapped = angles % (2 * np.pi)
    inputs = np.rint(wrapped / INPUT_SCALE + INPUT_ZERO_POINT)
    inputs = np.clip(inputs, -128, 127).astype(np.int8)
    inputs.tofile(build_dir / INPUT_FILE)
    return angles, expected

def read_output(build_dir: Path, num_points: int):
    output_file = build_dir / OUTPUT_FILE
    outputs = np.fromfile(output_file, dtype=np.int8)
    assert (outputs.size == num_points)
    return (outputs.astype(float) - OUTPUT_ZERO_POINT) * OUTPUT_SCALE

def plot_results(angles, expected, predictions):
    figure, axis = plt.subplots()
    axis.plot(angles, expected, label="sin(x)")
    axis.plot(angles, predictions, label="Model prediction")
    axis.set_xlabel("Input angle (radians)")
    axis.set_ylabel("Output")
    axis.grid(True)
    axis.legend()
    return figure

def test_hello_world(hw_target: str) -> None:
    build_dir = CWD / "build" / hw_target
    test_dir = build_dir / "hello_world"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    angles, expected = generate_input(test_dir)
    build_run_sim(CWD, build_dir, test_dir, hw_target)
    predictions = read_output(test_dir, len(angles))
    figure = plot_results(angles, expected, predictions)
    figure.savefig(CWD / PLOT_FILE.format(hw_target=hw_target))
    plt.close(figure)

    mse = float(np.mean((expected - predictions) ** 2))
    assert mse < MAX_MSE, f"MSE {mse:.6f} exceeds {MAX_MSE}"


if __name__ == "__main__":
    test_hello_world("native")
