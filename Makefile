build:
	(cd xtflm_interpreter && make install)

clean:
	(cd xtflm_interpreter && make clean)

init:
	python fetch_dependencies.py && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

test:
	(cd host_cmd_line_interpreter && make test)
	(cd xtflm_interpreter && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
