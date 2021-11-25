build:
	(cd tflm_interpreter && make install)

clean:
	(cd tflm_interpreter && make clean)

init:
	python3 fetch_dependencies.py

test:
	(cd host_cmd_line_interpreter && make test)
	(cd tflm_interpreter && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
