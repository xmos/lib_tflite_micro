.PHONY: patch build test

patch:
	(cd lib_tflite_micro/submodules/tflite-micro && git reset --hard && git apply ../../../patches/tflite-micro.patch)

build:
	cmake -B build
	make -j8 -C build

build_xs3:
	cmake -B build_xs3 --toolchain=lib_tflite_micro/submodules/xmos_cmake_toolchain/xs3a.cmake
	cmake --build build_xs3

build_vx4:
	cmake -B build_vx4 --toolchain=lib_tflite_micro/submodules/xmos_cmake_toolchain/vx4_xcc.cmake -DENABLE_SIZE_OPT=ON
	cmake --build build_vx4

test:
	(cd host_cmd_line_interpreter && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
