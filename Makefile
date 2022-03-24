build:
	(cd xtflm_interpreter && make install)

clean:
	(cd xtflm_interpreter && make clean)

init:
	python3 fetch_dependencies.py
	pip3 install --upgrade pip && \
	pip3 install -r requirements.txt -v -v -v

test:
	(. cd host_cmd_line_interpreter && make test)
	(. cd xtflm_interpreter && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
