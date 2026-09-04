patch:
	(cd lib_tflite_micro/submodules/tflite-micro && git reset --hard && git apply ../../../patches/tflite-micro.patch)

build:
	(cd lib_tflite_micro && ../version_check.sh)
	(cmake -B build && make -j8 -C build)

test:
	(cd host_cmd_line_interpreter && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
