.PHONY: clean clean-artifacts clean-dist test wheels dist release live-upload

clean: clean-artifacts clean-dist

clean-artifacts:
	rm -rf __pycache__ *.pyc *.pyo *.c *.html

clean-dist:
	rm -rf build/ dist/ wheelhouse/ fast_intensity.egg-info/ .eggs/

test:
	./run_tests.py

bench:
	./bench.py

visualize:
	cython -a fast_intensity.pyx
	open fast_intensity.html

wheels:
	./build-wheels.sh

dist:
	python setup.py sdist bdist_wheel

release:
	@echo "Building release..."
	$(MAKE) dist
	$(MAKE) wheels

live-upload:
	@echo "Pushing live release to pypi..."
	twine upload dist/*

# vim: ts=4
