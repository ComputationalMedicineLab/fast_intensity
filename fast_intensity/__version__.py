"""Fast intensity inference"""

# Single-sourced, canonical versioning for the package
# This file should never import anything from the package, it is instead
# imported by the package.

__version__ = '1.5.6'
__version_info__ = tuple(__version__.split('.'))

__author__ = 'Thomas A. Lasko, Jacek Bajor, John M Still'
__maintainer__ = 'John M Still'
__maintainer_email__ = 'john.m.still@vumc.org'

__all__ = ['__doc__',
           '__version__',
           '__version_info__',
           '__author__',
           '__maintainer__',
           '__maintainer_email__']
