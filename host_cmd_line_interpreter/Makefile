install:
	cmake -B build
	make install -j8 -C build

clean:
	rm -rf build

test: install
	make test -C tests
