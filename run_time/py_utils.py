#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  1 14:52:39 2018

@author: jim
"""
import numpy as np
from numba import jit
import collections
import threading
import scipy.ndimage.filters as scp_filters
import os
import sys

import gi

gi.require_version('Gtk', '3.0')

import PIL


def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def flatten(l):
    try:
        for el in l:
            if isinstance(el, collections.Iterable) and not isinstance(el, str):
                for sub in flatten(el):
                    yield sub
            else:
                yield el
    except TypeError:
        yield l


def uri_to_path(uri):
    return uri[7:]


def hex_to_rgb(hex_string):
    hex_string = hex_string.lstrip('#')
    lv = len(hex_string)
    return np.array([int(hex_string[i:i + lv // 3], 16) // 16 ** 2 for i in range(0, lv, lv // 3)])


def get_next_value_in_arr(arr, value, dir, current_index, axis=0, wrap=False):
    is_val_arr_equal = np.where(arr == value)[axis]
    if len(is_val_arr_equal) == 0:
        return None
    dist = (is_val_arr_equal - current_index) * dir
    if wrap:
        ind = np.where(dist < 0)
        dist[ind] = is_val_arr_equal[ind] * dir + arr.shape[axis] - current_index * dir
    else:
        ind = np.where(dist > 0)
        dist = dist[ind]
        if len(dist) == 0:
            return None
    return current_index + min(dist) * dir


def shift_cols_left(arr, step, val):
    ret = np.roll(arr, -step)
    ret[:, -step:] = np.matrix([val]).T
    return ret


def shift_cols_right(arr, step, val):
    ret = np.roll(arr, step)
    ret[:, :step] = np.matrix(val).T
    return ret


def get_local_path(path):
    try:
        wd = sys._MEIPASS
    except AttributeError:
        wd = os.getcwd()
    return os.path.join(wd, path)



@jit(cache=False, nopython=True, parallel=True)
def sigma(length, bound_cond):
    """
    Construct a sigma curve between two values in bound_cond = (lb, ub).
    :param length: int, length of array to be returned
    :param bound_cond:
    :return:
    """
    ran = np.arange(length)
    lb, ub = bound_cond
    A = np.log(1. / lb - 1)
    B = np.log(1. / ub - 1)
    alpha = (ran[-1] - ran[0]) / (A - B)
    shift = A * alpha + ran[0]
    return 1. / (1. + np.exp(-(ran - shift) / float(alpha)))


class FuncThread(threading.Thread):
    def __init__(self, target, *args):
        self._target = target
        self._args = args
        threading.Thread.__init__(self)

    def run(self):
        self._target(*self._args)


class ObjectWrapper(object):
    """
    Object wrapper class.
    This a wrapper for objects. It is initialized with the object to wrap
    and then proxies the unhandled getattribute methods to it.
    Other classes are to inherit from it.
    """

    def __init__(self, obj):
        '''
        Wrapper constructor.
        @param obj: object to wrap
        '''
        # wrap the object
        self._wrapped_obj = obj

    def __getattr__(self, attr):
        # see if this object has attr
        # NOTE do not use hasattr, it goes into
        # infinite recursion
        if attr in self.__dict__:
            # this object has it
            return getattr(self, attr)
        # proxy to the wrapped object
        return getattr(self._wrapped_obj, attr)


class ImageHandler(object):
    PILdic = {'gaussian_blur': PIL.ImageFilter.GaussianBlur,
              'min_filter': PIL.ImageFilter.MinFilter,
              'max_filter': PIL.ImageFilter.MaxFilter,
              'median_filter': PIL.ImageFilter.MedianFilter,
              'mode_filter': PIL.ImageFilter.MedianFilter}

    SCIPYdic = {'1d_blur': scp_filters.gaussian_filter1d,
                'laplace': scp_filters.laplace}

    @classmethod
    def get_support(cls):
        return list(cls.PILdic.items()) + list(cls.SCIPYdic.items())

    @classmethod
    def get_support_keys(cls):
        return list(cls.PILdic) + list(cls.SCIPYdic)

    @classmethod
    def blur(cls, img, size, typ='gaussian_blur'):
        cls.Filter(typ, img, size)

    @classmethod
    def Filter(cls, filt, img, size, axis=0, mode='const', cval=0):
        if filt in cls.PILdic:
            return np.ravel(np.array(PIL.Image.fromarray(img).filter(cls.PILdic[filt](size))))
        elif filt is '1d_blur':
            return np.ravel(cls.SCIPYdic['1d_blur'](img, size, axis=axis, mode=mode, cval=cval))
        elif filt is 'laplace':
            return np.ravel(cls.SCIPYdic['laplace'](img, mode=mode, cval=cval))

    @staticmethod
    def resize(arr, new_shape):
        return np.array(PIL.Image.fromarray(arr).resize(new_shape, resample=PIL.Image.NEAREST))
