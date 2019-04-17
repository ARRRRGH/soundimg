#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 22:08:09 2018

@author: jim
"""

import run_time.py_utils as py_utils
import numpy as np
from numba import jit


@jit(cache=False, nopython=True)
def fade_in_sigma(start, stop, init_value, end_value, samples, precision=1e-4):
    length = stop - start
    if length == 0:
        # print('fade in: length is 0 -> no fade in')
        return samples
    samples[start:stop] = py_utils.sigma(length, bound_cond=(precision, 1 - precision)) * (end_value - init_value) + init_value
    return samples


@jit(cache=False, nopython=True)
def fade_out_sigma(start, stop, init_value, end_value, samples,  precision=1e-4):
    length = stop - start
    if length == 0:
        # print('fade out: length is 0 -> no fade out')
        return samples
    samples[start:stop] = init_value - py_utils.sigma(length, bound_cond=(precision, 1 - precision)) * (init_value - end_value)
    return samples


@jit(cache=False, nopython=True)
def fade_in(start, stop, init_value, end_value, samples, precision=1e-4):
    """
    Linear fade in on array samples from indices start to stop and between values init_value and end_value. Returns
    changed samples.

    :param start:
    :param stop:
    :param init_value:
    :param end_value:
    :param samples:
    :param precision:
    :return:
    """
    length = stop - start
    if length == 0:
        #     print('fade in: length is 0 -> no fade in')
        return samples
    samples[start:stop] = np.linspace(init_value, end_value, length)
    # samples[start:stop] = init_value + fade_in_arr * (end_value - init_value)
    return samples


@jit(cache=False, nopython=True)
def fade_out(start, stop, init_value, end_value, samples, precision=1e-4):
    return fade_in(start, stop, init_value, end_value, samples, precision)
    # samples[start:stop] = end_value - fade_out_arr * (end_value - init_value)
    return samples


@jit(cache=False, nopython=True)
def blend(arr1, arr2):
    """
    Blend two arrays such that the resulting array transforms from arr1 to arr2. Uses a sigma decay

    :param arr1:
    :param arr2:
    :return:
    """
    transition = np.ones(arr2.shape)
    fade_out_sigma(0, arr1.shape[0], 1, 0, transition, 1e-6)
    return transition * arr1 + (1 - transition) * arr2


