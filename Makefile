.PHONY: patch build build_xs3 build_vx4 build_install clean test

JOBS := 16

build:
	cmake -G "Unix Makefiles" -B build
	cmake --build build --parallel $(JOBS)

build_xs3:
	cmake -G "Unix Makefiles" -B build_xs3 --toolchain=lib_tflite_micro/submodules/xmos_cmake_toolchain/xs3a.cmake
	cmake --build build_xs3 --parallel $(JOBS)

build_vx4:
	cmake -G "Unix Makefiles" -B build_vx4 --toolchain=lib_tflite_micro/submodules/xmos_cmake_toolchain/vx4_xcc.cmake
	cmake --build build_vx4 --parallel $(JOBS)

build_install:
	cmake --build build_xs3 --target project-install --parallel $(JOBS)
	cmake --build build_xs3 --target create_zip --parallel $(JOBS)

clean:
	rm -rf build build_xs3 build_vx4
	rm -f lib/*.a

test:
	cmake -S host_cmd_line_interpreter -B host_cmd_line_interpreter/build
	cmake --build host_cmd_line_interpreter/build --config Debug --parallel $(JOBS)
	ctest --test-dir host_cmd_line_interpreter/build -C Debug --output-on-failure
