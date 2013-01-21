#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from distutils.core import setup
from Cython.Build import cythonize
from Cython.Build.Dependencies import create_extension_list

modules = create_extension_list(['ceygen/*.pyx', 'ceygen/tests/*.pyx'])
for module in modules:
    module.language = "c++"

modules = cythonize(modules)
for module in modules:
    module.include_dirs.append('/usr/include/eigen3')
    module.extra_compile_args.append('-Wall')

setup(name='Ceygen',
      packages=['ceygen', 'ceygen.tests'],
      ext_modules=modules)
