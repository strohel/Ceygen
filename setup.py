#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from Cython.Build.Dependencies import create_extension_list

from distutils.core import setup

from support.dist import CeygenDistribution


modules = create_extension_list(['ceygen/*.pyx', 'ceygen/tests/*.pyx'])
for module in modules:
    module.language = "c++"

with open('README') as file:
    long_description = file.read()

setup(
    packages=['ceygen', 'ceygen.tests'],
    package_data={'ceygen': ['*.pxd']},
    distclass=CeygenDistribution,
    ext_modules=modules,
    include_dirs=['/usr/include/eigen3'],  # default overridable by setup.cfg
    cflags=['-O2', '-march=native'],  # ditto

    # meta-data; see http://docs.python.org/distutils/setupscript.html#additional-meta-data
    name='Ceygen',
    version="0.1",
    author='Matěj Laitl',
    author_email='matej@laitl.cz',
    maintainer='Matěj Laitl',
    maintainer_email='matej@laitl.cz',
    url='https://github.com/strohel/Ceygen',
    description='Cython helper for linear algebra with typed memoryviews built atop the Eigen C++ library',
    long_description=long_description,
    download_url='http://pypi.python.org/pypi/Ceygen',
    platforms='cross-platform',
    license='GNU GPL v2+',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Programming Language :: C++',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
