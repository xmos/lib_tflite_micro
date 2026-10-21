from pathlib import Path
import subprocess

CWD = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parent.parent
XMOS_CMAKE_TOOLCHAIN_DIR = ROOT_DIR / "lib_tflite_micro" / "submodules" / "xmos_cmake_toolchain"

TOOLCHAINS = {
    "xs3": "xs3a.cmake",
    "vx4-xcc": "vx4_xcc.cmake",
    "vx4-clang": "vx4_clang.cmake",
}

BUILD_DIRS = {
    "xs3": CWD / "build_xs3",
    "vx4-xcc": CWD / "build_vx4_xcc",
    "vx4-clang": CWD / "build_vx4_clang",
    "native": CWD / "build_native"
}

def test_cmake(hw_target: str) -> None:
    source_dir = Path(__file__).resolve().parent
    build_dir = BUILD_DIRS.get(hw_target, None)
    configure = ["cmake", "-S", str(source_dir), "-B", str(build_dir)]
    build = ["cmake", "--build", str(build_dir)]
    test = ["ctest", "--test-dir", str(build_dir), "--output-on-failure"]
    
    toolchain = TOOLCHAINS.get(hw_target, None)
    if toolchain is not None:
        toolchain_path = XMOS_CMAKE_TOOLCHAIN_DIR / toolchain   
        configure.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain_path}")

    subprocess.run(configure, check=True)
    subprocess.run(build, check=True)
    subprocess.run(test, check=True)
        
if __name__ == "__main__":
    test_cmake("native")
