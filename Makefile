PKG=fast_intensity
PKG_NAME=fast-intensity

purge: uninstall clean-caches clean-dist clean-artifacts

clean-artifacts:
	rm -rf **/__pycache__ **/*.pyc
	rm -f $(SRC)/{*.c,*.so}

clean-caches:
	rm -f .coverage*
	rm -rf .pytest_cache/ .cache/ .tox/ htmlcov/

clean-dist:
	rm -rf build/ dist/ wheelhouse/ $(PKG).egg-info/ .eggs/

install-dev:
	pip install -e .

uninstall:
	yes | pip uninstall $(PKG_NAME)

test:
	# Run tests just for currently most recent interpreter
	tox -e py37

test-each-env:
	tox -e py35,py36,py37

coverage:
	coverage erase
	tox -e py35-cov,py36-cov,py37-cov
	-coverage combine && coverage report

htmlcov: coverage
	coverage html
	cd htmlcov && python -m http.server

wheels:
	./build-wheels.sh

dist:
	python setup.py sdist bdist_wheel

release:
	@echo "Running tests..."
	tox -e py35,py36,py37
	@echo "Building release..."
	$(MAKE) dist
	$(MAKE) wheels

test-upload: release
	@echo "Uploading to test index..."
	twine upload -r testpypi dist/*

live-upload: release
	@echo "Pushing live release to pypi..."
	twine upload dist/*


.PHONY: purge clean-artifacts clean-caches clean-dist
.PHONY: install-dev uninstall
.PHONY: test test-each-env coverage htmlcov
.PHONY: wheels dist release test-upload live-upload

# vim: ts=4
