build:
	(cd xinterpreters/xinterpreters/host && make install)

clean:
	(cd xinterpreters/xinterpreters/host && make clean)

init:
	python3 fetch_dependencies.py
	python3 -m venv .venv && \
	. .venv/bin/activate && \
	pip3 install --upgrade pip && \
	pip3 install -r requirements.txt

test:
	(. .venv/bin/activate && pip3 install ./xinterpreters)
	(. .venv/bin/activate && cd host_cmd_line_interpreter && make test)
	(. .venv/bin/activate && cd xinterpreters/xinterpreters/host && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
