# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from distutils.command.build_ext import build_ext as orig_build_ext


class build_ext(orig_build_ext):

    user_options = orig_build_ext.user_options + [
        ('cflags=', None, "specify extra CFLAGS to pass to C and C++ compiler"),
        ('ldflags=', None, "specify extra LDFLAGS to pass to linker"),
    ]

    def initialize_options(self):
        orig_build_ext.initialize_options(self)
        self.cflags = None
        self.ldflags = None

    def finalize_options(self):
        orig_build_ext.finalize_options(self)
        if self.cflags is None:
            self.cflags = self.distribution.cflags or []
        if isinstance(self.cflags, str):
            self.cflags = self.cflags.split()

        if self.ldflags is None:
            self.ldflags = self.distribution.ldflags or []
        if isinstance(self.ldflags, str):
            self.ldflags = self.ldflags.split()

    def build_extension(self, ext):
        """HACK to actually apply cflags, ldflags"""
        orig_compile_args = ext.extra_compile_args
        ext.extra_compile_args = orig_compile_args or []
        ext.extra_compile_args.extend(self.cflags)
        orig_link_args = ext.extra_link_args
        ext.extra_link_args = orig_link_args or []
        ext.extra_link_args.extend(self.ldflags)

        orig_build_ext.build_extension(self, ext)

        ext.extra_compile_args = orig_compile_args
        ext.extra_link_args = orig_link_args
