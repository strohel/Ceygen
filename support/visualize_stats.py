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
    funcs = set([filename.split('-', 2)[0] for filename in files])
    for func in funcs:
        funcfiles = [filename for filename in files if fnmatch(filename, func + '-*')]
        natural_sort(funcfiles)
        sizes = None
        Z = []
        for filename in funcfiles:
            with open(filename) as f:
                contents = pickle.load(f)
            if sizes is None:
                sizes = contents['sizes']
            else:
                assert sizes == contents['sizes']
            Z.append(contents['stats'])

        yticklabels = [splitext(filename)[0][len(func) + 1:] for filename in funcfiles]
        fig = plt.figure()
        fig.canvas.manager.set_window_title(func)
        X = range(len(sizes))
        Y = range(len(yticklabels))
        ax = fig.add_subplot(111, xlabel='size', xticks=X, xticklabels=sizes, yticks=Y,
                yticklabels=yticklabels, zlabel="GFLOPS", projection='3d')
        X, Y = np.meshgrid(X, Y)
        surf = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    plt.show()

if __name__ == '__main__':
    main()
