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
