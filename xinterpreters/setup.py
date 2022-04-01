import setuptools

LIB_XINTERPRETERS = [
    "host/libs/linux/xtflm_python.so",
    "host/libs/linux/xtflm_python.so.1.0.1",
    "host/libs/macos/xtflm_python.dylib",
    "host/libs/macos/xtflm_python.1.0.1.dylib",
]

EXCLUDES = ["host/CMakeLists.txt", "host/build", "host/tests", "host/src"]

INSTALL_REQUIRES = [
    "numpy<2.0",
    "portalocker==2.0.0",
]

setuptools.setup(
    name="xinterpreters",
    packages=setuptools.find_packages(exclude=EXCLUDES),
    python_requires=">=3.8.0",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "test": [
            "pytest>=5.2.0",
        ],
    },
    package_data={"": LIB_XINTERPRETERS},
    author="XMOS",
    author_email="support@xmos.com",
    description="XMOS Host and Device Interpreters.",
    license="LICENSE.txt",
    keywords="xmos xcore",
    use_scm_version={
        "root": "..",
        "relative_to": __file__,
        "version_scheme": "post-release",
    },
    setup_requires=["setuptools_scm"],
)