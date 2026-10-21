from pathlib import Path
import subprocess


def test_cmake(cmake_build_dir: Path) -> None:
    subprocess.run(
        ["cmake", "--build", str(cmake_build_dir), "--target", "avg_pool"],
        check=True,
    )
    subprocess.run(
        [
            "ctest", "--test-dir", str(cmake_build_dir),
            "--output-on-failure", "--no-tests=error", "-R", "^avg_pool$",
        ],
        check=True,
    )
