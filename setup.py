"""
Fast intensity inference
"""
from codecs import open
from distutils.version import LooseVersion
from os.path import abspath, dirname, join

from setuptools import setup, Extension

MIN_CYTHON = '0.28.0'
NAME = 'fast-intensity'
BASE_DIR = abspath(dirname(__file__))

# set up the setup() metadata
metadata = dict(
    name=NAME,
    description=__doc__.strip(),
    url='https://github.com/ComputationalMedicineLab/fast_intensity',
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
)

version_path = join(BASE_DIR, 'fast_intensity', '__version__.py')
with open(version_path, encoding='utf-8') as fd:
    # loads a bunch of metadata
    data = {}
    exec(fd.read(), {}, data)
    for key in ['version', 'author', 'maintainer', 'maintainer_email']:
        metadata[key] = data['__{}__'.format(key)]
    del data

with open(join(BASE_DIR, 'README.md'), encoding='utf-8') as fd:
    metadata['long_description'] = fd.read()
    metadata['long_description_content_type'] = 'text/markdown'

# Load build-time dependencies
try:
    import numpy
except ImportError:
    msg = '{} requires numpy to build'
    raise ImportError(msg.format(NAME))
else:
    np_includes = [numpy.get_include()]

try:
    import Cython
    assert Cython.__version__ >= LooseVersion(MIN_CYTHON)
except (ImportError, AssertionError):
    msg = '{} requires cython>={} to build'
    raise ImportError(msg.format(NAME, MIN_CYTHON))
else:
    from Cython.Build import build_ext as cython_build_ext

# Run the build
setup(
    **metadata,

    # Install data
    zip_safe=False,
    include_package_data=True,
    packages=['fast_intensity'],
    # Cython is not required a runtime, it is not an installation dependency
    install_requires=[
        'numpy',
        'scipy',
    ],
    test_suite='tests',
    ext_modules=[
        Extension(
            name="fast_intensity.stair_step",
            sources=["fast_intensity/stair_step.pyx"],
            include_dirs=np_includes,
        ),
        Extension(
            name="fast_intensity.fast_hist",
            sources=["fast_intensity/fast_hist.pyx"],
            include_dirs=np_includes,
        ),
    ],
    cmdclass={
        'build_ext': cython_build_ext,
    },
)
