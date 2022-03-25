build:
	(cd xtflm_interpreter && make install)

clean:
	(cd xtflm_interpreter && make clean)

init:
	which python3
	python3 --version
	PYENV_DEBUG=1 python3 --version
	
	pyenv local 3.7.12 && \
	python -m venv .venv && \
	. .venv/bin/activate && \
	python fetch_dependencies.py && \
	which python && \
	python --version && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

test:
	. .venv/bin/activate && \
	(cd host_cmd_line_interpreter && make test)
	(cd xtflm_interpreter && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
