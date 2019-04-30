.PHONY: clean clean-artifacts clean-dist test wheels dist release live-upload

clean: clean-artifacts clean-dist

clean-artifacts:
	rm -rf __pycache__ *.pyc *.pyo *.c

clean-dist:
	rm -rf build/ dist/ wheelhouse/ fast_intensity.egg-info/ .eggs/

test:
	python -m test_fast_intensity -vv

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
