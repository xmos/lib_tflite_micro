patch:
	(cd lib_tflite_micro/submodules/tflite-micro && git reset --hard && git apply --directory tensorflow ../../../patches/tflite-micro.patch)

build:
	(cd lib_tflite_micro && ../version_check.sh)
	mkdir -p build
	(cd build && cmake .. && make -j8)

init:
	python3 fetch_dependencies.py
	python3 -m venv .venv && \
	. .venv/bin/activate && \
	pip3 install --upgrade pip && \
	pip3 install -r requirements.txt

test:
	(. .venv/bin/activate && cd host_cmd_line_interpreter && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
