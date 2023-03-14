build:
	(cd lib_tflite_micro && ../version_check.sh)
	(cd lib_tflite_micro/submodules/tflite-micro && git reset --hard && patch -p0 -i ../../../patches/tflite-micro.patch)
  	(cd lib_tflite_micro/submodules/gemmlowp && git reset --hard && patch -p0 -i ../../../patches/gemmlowp.patch)
	mkdir -p tflite_micro_compiler/build
	(cd tflite_micro_compiler/build && cmake .. -DXBUILD=1 && make -j8)

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
