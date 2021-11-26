# Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
# XMOS Public License: Version 1
import setuptools

EXCLUDES = ["README.rst"]

INSTALL_REQUIRES = [
]

setuptools.setup(
    name="flash_builder",
    packages=setuptools.find_packages(exclude=EXCLUDES),
    python_requires=">=3.8.0",
    install_requires=INSTALL_REQUIRES,
    extras_require={},
    package_data={},
    author="XMOS",
    author_email="support@xmos.com",
    description="XMOS Flash Builder for TensorFlow Lite model interpreter.",
    license="LICENSE.txt",
    keywords="xmos xcore",
    use_scm_version={
        "root": "..",
        "relative_to": __file__,
        "version_scheme": "post-release",
    },
    setup_requires=["setuptools_scm"],
)
