patch:
	(cd lib_tflite_micro/submodules/tflite-micro && git reset --hard && git apply ../../../patches/tflite-micro.patch)

build:
	(cd lib_tflite_micro && ../version_check.sh)
	(cmake -B build && make -j8 -C build)

init:
	python3 fetch_dependencies.py
	pip3 install -r requirements.txt

test:
	(cd host_cmd_line_interpreter && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""