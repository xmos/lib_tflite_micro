.PHONY: patch build build_xs3 build_vx4 build_install clean test

JOBS := $(shell nproc --ignore=1)

patch:
	(cd lib_tflite_micro/submodules/tflite-micro && git reset --hard && git apply ../../../patches/tflite-micro.patch)

build:
	cmake -B build
	make -j$(JOBS) -C build

build_xs3:
	cmake -B build_xs3 --toolchain=lib_tflite_micro/submodules/xmos_cmake_toolchain/xs3a.cmake
	cmake --build build_xs3 --parallel $(JOBS)

build_vx4:
	cmake -B build_vx4 --toolchain=lib_tflite_micro/submodules/xmos_cmake_toolchain/vx4_xcc.cmake
	cmake --build build_vx4 --parallel $(JOBS)

build_install:
	cmake --build build_xs3 --target project-install --parallel $(JOBS)
	cmake --build build_xs3 --target create_zip --parallel $(JOBS)

clean:
	rm -rf build build_xs3 build_vx4
	rm -f lib/*.a

test:
	(cd host_cmd_line_interpreter && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
