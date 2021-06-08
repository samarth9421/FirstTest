# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 01:43:05 2021

@author: Team A8
"""


# import io
import os
from pathlib import Path

from setuptools import find_packages, setup


# Package meta-data.
NAME = 'credit_risk'
DESCRIPTION = 'Calculates credit risk and gives interpreted DNN graphs'
AUTHOR = 'Team A8'
REQUIRES_PYTHON = '>=3.8.0'


def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()



here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
# try:
#     with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
#         long_description = '\n' + f.read()
# except FileNotFoundError:
#     long_description = DESCRIPTION


ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
about = {}
# with open(PACKAGE_DIR / 'VERSION') as f:
#     _version = f.read().strip()
#     about['__version__'] = _version


setup(
    name=NAME,
    version='0.1.0',
    description=DESCRIPTION,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=('tests',)),
    install_requires=list_reqs(),
    extras_require={},
    package_data={'credit_risk': ['save_files/datasets/test_data.csv',
                                  'save_files/datasets/train_data.csv',
                                  'save_files/lime_estimator/*.pkl',
                                  'save_files/trained_models/*.pkl',
                                  'save_files/untrained_models/*.pkl']},
    include_package_data=True,

)
