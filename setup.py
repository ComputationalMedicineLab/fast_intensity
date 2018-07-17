"""
Fast intensity inference
"""
from codecs import open
from distutils.version import LooseVersion
from os.path import abspath, dirname, join

from setuptools import setup, Extension

# dependency versions
min_cython = '0.28.0'

# Metadata
name = 'fast-intensity'
package = 'fast_intensity'
version = '0.1.5'
description = __doc__.strip()
author = 'Thomas A. Lasko, Jacek Bajor'
maintainer = 'John M Still'
maintainer_email = 'john.m.still@vumc.org'
url = 'https://github.com/ComputationalMedicineLab/{}'.format(package)
license = 'BSD'
classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Information Analysis',
]

with open(join(abspath(dirname(__file__)), 'README.md'),
          encoding='utf-8') as fd:
    long_description = fd.read()

# Check build-time dependencies
try:
    import numpy
except ImportError:
    msg = '{} requires numpy to build'
    raise ImportError(msg.format(name))
else:
    np_includes = [numpy.get_include()]

try:
    import Cython
    assert Cython.__version__ >= LooseVersion(min_cython)
except (ImportError, AssertionError):
    msg = '{} requires cython>={} to build'
    raise ImportError(msg.format(name, min_cython))
else:
    from Cython.Build import build_ext as cython_build_ext

# Setup the extensions
ext_modules = [
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
]

# Cython is only required to build, not at runtime
install_requires = ['numpy', 'scipy']

# Run the build
setup(
    # Metadata
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=author,
    maintainer=maintainer,
    maintainer_email=maintainer_email,
    url=url,
    license=license,
    classifiers=classifiers,

    # Install data
    zip_safe=False,
    include_package_data=True,
    packages=['fast_intensity'],
    install_requires=install_requires,
    test_suite='tests',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': cython_build_ext,
    },
)
