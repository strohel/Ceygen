#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2013 Matěj Laitl <matej@laitl.cz>
# Distributed under the terms of the GNU General Public License v2 or any
# later version of the license, at your option.

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from fnmatch import fnmatch
from glob import glob
from os.path import splitext
import pickle
import re


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return l.sort(key=alphanum_key)

def main():
    files = glob('*-*.pickle')
    #funcs = set([filename.split('.', 2)[0] for filename in files])
    funcs = ('add_vv', 'det', 'dot_mm', 'dot_mv')
    markers = ('x', 'o', 's', 'v')
    linestyles = ('-', '--', '-.', ':')

    maxsizeindex = 13
    for func in funcs:
        funcfiles = [filename for filename in files if fnmatch(filename, func + '.*')]
        natural_sort(funcfiles)
        sizes = None
        Z = []
        for filename in funcfiles:
            with open(filename) as f:
                contents = pickle.load(f)
            if sizes is None:
                sizes = contents['sizes'][0:maxsizeindex]
            else:
                assert sizes == contents['sizes'][0:maxsizeindex]
            Z.append(contents['percall'][0:maxsizeindex])

        variants = [filename.split('.')[1].split('-')[0] for filename in funcfiles]
        fig = plt.figure()
        fig.canvas.manager.set_window_title(func)
        X = range(len(sizes))

        # compute relative times
        for i in range(len(sizes)):
            maximum = max((z[i] for z in Z))
            for z in Z:
                z[i] /= maximum

        ax = fig.add_subplot(111, xlabel='matrix/vector size (one side)', xticks=X, xticklabels=sizes,
                             ylabel="relative time per call", title=func)
        for (variant, y, marker, linestyle) in zip(variants, Z, markers, linestyles):
            ax.plot(X, y, label=variant, marker=marker, linestyle=linestyle)
        ax.set_ylim((0, 1.1))
        ax.legend(loc=0)
    plt.show()

if __name__ == '__main__':
    main()
