build:
	(cd xinterpreters/xinterpreters/host && make install)

clean:
	(cd xinterpreters/xinterpreters/host&& make clean)

init:
	python fetch_dependencies.py

test:
	(cd host_cmd_line_interpreter && make test)
	(cd xinterpreters && python3 xinterpreters_test.py)
	(cd xinterpreters/xinterpreters/host && make test)
	@echo ""
	@echo "All tests PASS"
	@echo ""
