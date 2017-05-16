from setuptools import setup, find_packages
from codecs import open
from distutils.core import setup
from distutils.extension import Extension
from os import path

from Cython.Build import cythonize
import numpy as np

__version__ = '0.1.1'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

extensions = [
    Extension("fast_intensity/stair_step", ["fast_intensity/stair_step.pyx"],
        include_dirs = [np.get_include()]),
    Extension("fast_intensity/fast_hist", ["fast_intensity/fast_hist.pyx"],
        include_dirs = [np.get_include()])
]

setup(
    name='fast-intensity',
    version=__version__,
    description='Fast density inference',
    long_description=long_description,
    url='https://github.com/ComputationalMedicineLab/fast_intensity',
    download_url='https://github.com/ComputationalMedicineLab/fast_intensity/tarball/v' + __version__,
    license='BSD',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
      'License :: OSI Approved :: BSD License',
      'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author='Thomas A. Lasko, Jacek Bajor',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='jacek.m.bajor@vanderbilt.edu',
    test_suite='tests',
    ext_modules = cythonize(extensions),
)
