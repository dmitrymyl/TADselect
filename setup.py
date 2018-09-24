#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import io
import os
import re

classifiers = """\
    Development Status :: 1 - Planning
    Operating System :: Unix
    Programming Language :: Python
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.4
    Programming Language :: Python :: 3.5
    Programming Language :: Python :: 3.6
"""

def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop('encoding', 'utf-8')
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text

def get_version():
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        _read('TADselect', '__init__.py'),
        re.MULTILINE).group(1)
    return version

def get_long_description():
    return _read('README.md')

requirements = [
                'numpy>=1.12.0',
                'pandas>=0.22.0',
                'cooler>=0.7.9',
                'lavaburst>=0.2.0'
                ]

setup(
    name='TADselect',
    version=get_version(),
    description="Python library to run TAD callers",
    long_description=get_long_description(),
    keywords=['genomics', 'bioinformatics', 'Hi-C', 'TAD'],
    author="Dmitry Mylarcshikov & Aleksandra Galitsyna",
    author_email='agalitzina@gmail.com',
    url='https://github.com/agalitsyna/TADselect',

    license="BSD",
    include_package_data=True,

    install_requires=requirements,
    zip_safe=False,

    test_suite='tests',

    packages=['TADselect'],
    package_dir={'TADselect': 'TADselect'},

    classifiers=[s.strip() for s in classifiers.split('\n') if s],
)
