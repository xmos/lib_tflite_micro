install:
	mkdir -p build
	(cd build && cmake .. && make install)

clean:
	rm -rf build

test: install
	(cd tests && make test)
