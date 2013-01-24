# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from distutils import log
from distutils.command.clean import clean as orig_clean
from os import remove
from os.path import exists, splitext


class clean(orig_clean):

    def run(self):
        orig_clean.run(self)
        for module in self.distribution.ext_modules:
            # following fails if one says ./setup.py build clean, but it's a corner case
            if module.sources and module.sources[0].endswith('.pyx'):
                base = splitext(module.sources[0])[0]
                for ext in ('.cpp', '.html'):
                    filename = base + ext
                    if exists(filename):
                        log.debug("Removing {0}".format(filename))
                        remove(filename)
