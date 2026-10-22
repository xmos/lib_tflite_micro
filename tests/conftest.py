from pathlib import Path
import subprocess

import pytest


ROOT_DIR = Path(__file__).resolve().parent.parent
XMOS_CMAKE_TOOLCHAIN_DIR = ROOT_DIR / "lib_tflite_micro" / "submodules" / "xmos_cmake_toolchain"

TOOLCHAINS = {
    "xs3": "xs3a.cmake",
    "vx4-xcc": "vx4_xcc.cmake",
    "vx4-clang": "vx4_clang.cmake",
}


def build_run_sim(source_dir: Path, build_dir: Path, test_dir: Path, hw_target: str):
    configure = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    build = ["cmake", "--build", str(build_dir)]
    test = ["ctest", "--test-dir", str(test_dir), "--output-on-failure", "--no-tests=error"]
    tch = TOOLCHAINS.get(hw_target, None)
    if tch is not None:
        toolchain_path = XMOS_CMAKE_TOOLCHAIN_DIR / tch
        configure.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain_path}")
    subprocess.run(configure, check=True)
    subprocess.run(build, check=True)
    subprocess.run(test, check=True)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--hw-target",
        choices=("xs3", "native", "vx4-xcc", "vx4-clang"),
        default="native",
        help="Build and CTest target (default: native); activate XMOS tools first.",
    )


@pytest.fixture(scope="session")
def hw_target(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--hw-target")
