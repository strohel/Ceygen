# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from distutils.dist import Distribution

from .dist_cmd_build_ext import build_ext
from .dist_cmd_test import test


class CeygenDistribution(Distribution):

    def __init__(self, attrs):
        self.cflags = None  # Default CFLAGS overridable by setup.cfg
        self.ldflags = None  # Default LDFLAGS overridable by setup.cfg
        Distribution.__init__(self, attrs)
        self.cmdclass['build_ext'] = build_ext
        self.cmdclass['test'] = test
