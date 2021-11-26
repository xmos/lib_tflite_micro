build:
	(cd tflm_interpreter && make install)

clean:
	(cd tflm_interpreter && make clean)

init:
	python3 fetch_dependencies.py
	python3 -m venv .venv && \
	. .venv/bin/activate && \
	pip3 install -r requirements.txt

test:
	(. .venv/bin/activate && cd host_cmd_line_interpreter && make test)
	(. .venv/bin/activate && cd tflm_interpreter && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
