from pathlib import Path

import pytest

from conftest import build_run_sim

CWD = Path(__file__).resolve().parent

def test_avg_pool(hw_target: str):
    if hw_target == "vx4-clang":
        pytest.xfail("VX4 Clang build and simulator support is not ready")
    build_dir = CWD / "build" / hw_target
    test_dir = build_dir / "avg_pool_test"
    build_run_sim(CWD, build_dir, test_dir, hw_target)    
    
if __name__ == "__main__":
    test_avg_pool("native")
