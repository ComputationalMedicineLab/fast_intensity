"""
Fast intensity inference
"""
from os.path import abspath, dirname, join

from setuptools import setup, Extension
import numpy
from Cython.Build import cythonize

readme_path = join(abspath(dirname(__file__)), 'README.md')
with open(readme_path, encoding='utf-8') as file:
    DESCRIPTION = file.read()

extensions = [
    Extension(
        'fast_intensity',
        ['fast_intensity.pyx'],
        include_dirs=[numpy.get_include()],
    )
]

# Run the build
setup(
    name='fast-intensity',
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
    version='0.4',
    author='Thomas A. Lasko, Jacek Bajor, John M Still',
    maintainer='John M Still',
    maintainer_email='john.m.still@vumc.org',
    long_description=DESCRIPTION,
    long_description_content_type='text/markdown',
    # Install data
    zip_safe=False,
    include_package_data=True,
    # Cython is not required a runtime, it is not an installation dependency
    install_requires=['numpy', 'scipy'],
    ext_modules=cythonize(extensions),
)
