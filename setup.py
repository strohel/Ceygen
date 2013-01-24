#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from Cython.Build import cythonize
from Cython.Build.Dependencies import create_extension_list

from distutils.cmd import Command
from distutils.core import setup
from distutils.errors import DistutilsExecError

from os.path import abspath, dirname, join
import sys
import unittest

modules = create_extension_list(['ceygen/*.pyx', 'ceygen/tests/*.pyx'])
for module in modules:
    module.language = "c++"

modules = cythonize(modules)
for module in modules:
    module.include_dirs.append('/usr/include/eigen3')
    module.extra_compile_args.append('-Wall')

class test(Command):
    """Test Ceygen in the build directory"""

    description = 'run unit test-suite of Ceygen within the build directory'
    user_options = []

    def initialize_options(self):
        self.build_lib = None

    def finalize_options(self):
        self.set_undefined_options('build', ('build_lib', 'build_lib'))

    def run(self):
        self.run_command('build')  # build if not alredy run
        orig_path = sys.path[:]
        try:
            build_path = abspath(self.build_lib)
            sys.path.insert(0, build_path)
            import ceygen.tests as t
            assert dirname(t.__file__) == join(build_path, 'ceygen', 'tests')
            suite = unittest.TestLoader().loadTestsFromModule(t)
            result = unittest.TextTestRunner(verbosity=self.verbose).run(suite)
            if not result.wasSuccessful():
                raise Exception("There were test failures")
        except Exception as e:
            raise DistutilsExecError(e)
        finally:
            sys.path = orig_path

setup(
    packages=['ceygen', 'ceygen.tests'],
    package_data={'ceygen': ['*.pxd']},
    cmdclass = {'test': test},
    ext_modules=modules,

    # meta-data; see http://docs.python.org/distutils/setupscript.html#additional-meta-data
    name='Ceygen',
    version="0.1",
    author=u'Matěj Laitl',
    author_email='matej@laitl.cz',
    maintainer=u'Matěj Laitl',
    maintainer_email='matej@laitl.cz',
    url='https://github.com/strohel/Ceygen',
    description='Cython helper for linear algebra with typed memoryviews built atop of Eigen C++ library',
    long_description='Ceygen is a binary Python extension module helper for linear ' +
        'algebra with Cython typed memoryviews. Cython is built atop of Eigen C++ ' +
        'library. Ceygen is not a Cython wrapper or an interface to Eigen!',
    # Note to myself: must manually upload on each release!
    #download_url='https://github.com/downloads/strohel/Ceygen/Ceygen-'+version+'.tar.gz',
    platforms='cross-platform',
    license='GNU GPL v2+',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
