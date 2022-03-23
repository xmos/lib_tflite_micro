build:
	(cd xtflm_interpreter && make install)

clean:
	(cd xtflm_interpreter && make clean)

init:
	python3 fetch_dependencies.py
	python3 -m venv .venv && \
	. .venv/bin/activate && \
	python -m pip install pip==22.0.4 && \
	pip3 install -r requirements.txt

test:
	(. .venv/bin/activate && cd host_cmd_line_interpreter && make test)
	(. .venv/bin/activate && cd xtflm_interpreter && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
