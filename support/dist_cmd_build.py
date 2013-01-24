# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from Cython.Build import cythonize

from distutils.command.build import build as orig_build


class build(orig_build):

    def run(self):
        self.distribution.ext_modules = cythonize(self.distribution.ext_modules,
                annotate=True, force=self.force)
        orig_build.run(self)
