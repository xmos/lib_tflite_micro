from pathlib import Path
import subprocess

import pytest


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


@pytest.fixture(scope="session")
def cmake_build_dir(hw_target: str) -> Path:
    source_dir = Path(__file__).resolve().parent
    toolchain_dir = (
        source_dir.parent / "lib_tflite_micro" / "submodules" / "xmos_cmake_toolchain"
    )
    toolchains = {
        "xs3": "xs3a.cmake",
        "vx4-xcc": "vx4_xcc.cmake",
        "vx4-clang": "vx4_clang.cmake",
    }
    build_dirs = {
        "xs3": "build_xs3",
        "vx4-xcc": "build_vx4_xcc",
        "vx4-clang": "build_vx4_clang",
        "native": "build_native",
    }
    build_dir = source_dir / build_dirs[hw_target]
    configure = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    if hw_target in toolchains:
        toolchain_path = toolchain_dir / toolchains[hw_target]
        configure.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain_path}")

    subprocess.run(configure, check=True)
    return build_dir
