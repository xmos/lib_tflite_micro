from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np


INPUT_FILE = "input.bin"
OUTPUT_FILE = "output.bin"
PLOT_FILE = "comparison.png"
MAX_MSE = 0.01

# Quantization parameters of the checked-in generated model.
INPUT_SCALE = 0.024480115622282028
INPUT_ZERO_POINT = -128
OUTPUT_SCALE = 0.0082909567281603813
OUTPUT_ZERO_POINT = 5


def generate_input(build_dir: Path):
    inputs = np.arange(-128, 128, dtype=np.int8)
    angles = (inputs.astype(float) - INPUT_ZERO_POINT) * INPUT_SCALE
    inputs.tofile(build_dir / INPUT_FILE)
    return angles, np.sin(angles)


def run_model(build_dir: Path, num_points: int):
    output_file = build_dir / OUTPUT_FILE
    output_file.unlink(missing_ok=True)
    subprocess.run(
        [
            "ctest", "--test-dir", str(build_dir),
            "--output-on-failure", "--no-tests=error", "-R", "^hello_world$",
        ],
        check=True,
    )
    outputs = np.fromfile(output_file, dtype=np.int8)
    assert outputs.size == num_points, (
        f"Expected {num_points} output bytes, got {outputs.size}"
    )
    return (outputs.astype(float) - OUTPUT_ZERO_POINT) * OUTPUT_SCALE


def plot_results(build_dir: Path, angles, expected, predictions, mse: float) -> None:
    figure, axis = plt.subplots()
    axis.plot(angles, expected, label="sin(x)")
    axis.plot(angles, predictions, label="Model prediction")
    axis.set_xlabel("Input angle (radians)")
    axis.set_ylabel("Output")
    axis.set_title(f"Hello World: MSE = {mse:.6f}")
    axis.grid(True)
    axis.legend()
    figure.savefig(build_dir / PLOT_FILE)
    plt.close(figure)


def test_hello_world(cmake_build_dir: Path) -> None:
    subprocess.run(
        ["cmake", "--build", str(cmake_build_dir), "--target", "hello_world"],
        check=True,
    )
    build_dir = cmake_build_dir / "hello_world"
    angles, expected = generate_input(build_dir)
    predictions = run_model(build_dir, len(angles))
    mse = float(np.mean((expected - predictions) ** 2))
    plot_results(build_dir, angles, expected, predictions, mse)

    print(f"Hello World MSE: {mse:.6f}")
    assert mse < MAX_MSE, f"MSE {mse:.6f} exceeds {MAX_MSE}"
