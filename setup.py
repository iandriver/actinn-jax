#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os
import codecs
import sys

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

'''
with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()
'''
requirements = [
    "numpy>=1.15",
    "scipy>=1.7",
    "anndata>=0.6.22",
    "pandas>=1.3",
    "h5py",
    "scanpy>=1.8",
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    "optax>=0.1.5",
]

extras_require = {
    # Apple-silicon GPU acceleration (optional): pip install .[metal]
    "metal": ["jax-metal"],
    "test": ["pytest"],
}


author = 'Ian Driver'

setup(
    author=author,
    author_email='driver.ian@gmail.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    description="Fast JAX reimplementation of ACTINN for single-cell cell-type reference mapping",
    python_requires='>=3.10',
    install_requires=requirements,
    extras_require=extras_require,
    license="MIT license",
    include_package_data=True,
    keywords='single-cell cell-type-annotation ACTINN jax scanpy',
    name='actinn-jax',
    packages=find_packages(),
    package_dir={'actinn_jax':
                 'actinn_jax'},
    version=get_version("actinn_jax/__init__.py"),
    zip_safe=False,
)