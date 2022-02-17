# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
import setuptools

LIB_XTFLM_INTERPRETER = [
    "libs/linux/xtflm_python.so",
    "libs/linux/xtflm_python.so.1.0.1",
    "libs/macos/xtflm_python.dylib",
    "libs/macos/xtflm_python.1.0.1.dylib",
]

EXCLUDES = ["CMakeLists.txt", "README.rst", "build", "tests", "example.py", "goldfish.png", "src"]

INSTALL_REQUIRES = [
    "numpy<2.0",
    "portalocker==2.0.0",
]

setuptools.setup(
    name="xtflm_interpreter",
    packages=setuptools.find_packages(exclude=EXCLUDES),
    python_requires=">=3.8.0",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "test": [
            "pytest>=5.2.0",
        ],
    },
    package_data={"": LIB_XTFLM_INTERPRETER},
    author="XMOS",
    author_email="support@xmos.com",
    description="XMOS TensorFlow Lite model interpreter.",
    license="LICENSE.txt",
    keywords="xmos xcore",
    use_scm_version={
        "root": "..",
        "relative_to": __file__,
        "version_scheme": "post-release",
    },
    setup_requires=["setuptools_scm"],
)
