#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='v1_likelihood',
    version='0.0.0',
    description='V1 likelihood analysis',
    author='Edgar Y. Walker',
    author_email='eywalker@bcm.edu',
    packages=find_packages(exclude=[]),
    install_requires=['numpy', 'tqdm', 'gitpython', 'python-twitter', 'scikit-image', 'datajoint', 'attorch', 'h5py'],
)
