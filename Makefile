build:
	(cd xinterpreters/host && make install)

clean:
	(cd xinterpreters/host&& make clean)

init:
	python fetch_dependencies.py && \
	pip install --upgrade pip && \
	pip install -r requirements.txt

test:
	(cd host_cmd_line_interpreter && make test)
	python xinterpreters_test.py
	(cd xinterpreters/host && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
