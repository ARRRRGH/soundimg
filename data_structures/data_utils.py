import run_time.config as config
import run_time.py_utils as py_utils

import ctypes
from abc import ABCMeta, abstractmethod
import multiprocess as mp
import numpy as np


class InitialConditions(object):
    def __init__(self, img_shape, track):

        self.shape = img_shape
        self.track = track
        self.nr_part_waves_in_timbre = self.track.instrument.timbre.shape[0]

        self.inner_phases = None
        self.outer_phases = None
        self.last_freq_spectrs = None
        self.is_init_condition_set = None
        self._inits_currently_reset = False

        self.initialize(self.nr_part_waves_in_timbre)

    def get(self, where):
        return tuple(self.get_per_channel(i, where) for i in range(config.params.channels))

    def get_per_channel(self, channel, where):
        return self.outer_phases[where, channel], self.inner_phases[:, where, :, channel], \
               self.last_freq_spectrs[:, where, channel]

    def is_currently_reset(self):
        return self._inits_currently_reset

    def initialize(self, nr_part_waves_in_timbre):
        self.outer_phases = np.zeros((self.shape[1], config.params.channels))
        self.inner_phases = np.zeros((self.shape[0], self.shape[1], nr_part_waves_in_timbre, config.params.channels))
        self.last_freq_spectrs = np.zeros((self.shape[0], self.shape[1], config.params.channels))

        self.is_init_condition_set = np.zeros(self.shape[1])
        self._inits_currently_reset = True

    def set_per_channel(self, channel, where, conditions):
        self.outer_phases[where, channel] = conditions[0]
        self.inner_phases[:, where, :, channel] = conditions[1]
        self.last_freq_spectrs[:, where, channel] = conditions[2]

    def set(self, conditions, where):
        for channel in range(len(conditions)):
            self.set_per_channel(channel, where, conditions[channel])

        self.is_init_condition_set[where] = 1
        self._inits_currently_reset = False

    def reset(self):
        if self.nr_part_waves_in_timbre == self.track.instrument.nr_of_part_waves:
            self.outer_phases[:, :] = 0
            self.inner_phases[:, :, :, :] = 0
            self.last_freq_spectrs[:, :, :] = 0

            self.is_init_condition_set[:] = 0
            self._inits_currently_reset = True
        else:
            self.nr_part_waves_in_timbre = self.track.instrument.nr_of_part_waves
            self.initialize(self.nr_part_waves_in_timbre)

    def get_last(self, px_range):
        current = px_range.start % self.shape[1]
        times = px_range.start // self.shape[1]

        if self.is_init_condition_set[current]:
            return self.get(current), px_range.start
        else:
            # fall back on last known init_condition
            last = py_utils.get_next_value_in_arr(self.is_init_condition_set, 1, -1, current, wrap=False)

            # if there is no init_condition set (when starting generation of image), return initial init_cond
            # and don't shift the range to be generated
            if last is None:
                return self.get(0), px_range.start

            # if there is an earlier init_cond set, use this one and adapt the range to be generated
            else:
                return self.get(last), last + times * self.shape[1]


class Buffer(object):
    def __init__(self, channels, length, dtype, shared):
        self.channels = channels
        assert self.channels == 1 or self.channels == 2

        self.loop_arr = LoopArray(np.zeros(int(length * self.channels), dtype=dtype), dtype=dtype, shared=shared,
                                  modes=('wrap', 'raise'))

    @property
    def arr(self):
        return self.loop_arr.arr

    @property
    def arr_left(self):
        return self.loop_arr.arr[::2]

    @property
    def arr_right(self):
        return self.loop_arr.arr[1::2]

    def reset(self):
        self.loop_arr.arr[:] = 0

    def _enlarge_slice_or_int_key(self, key, factor):
        if isinstance(key, slice):
            return slice(factor * key.start, factor * key.stop, None)
        elif isinstance(key, int):
            return slice(key, key + factor)

    def _double_key(self, key):
        if not isinstance(key, np.ndarray):
            return self._enlarge_slice_or_int_key(key, 2)
        else:
            shape = list(key.shape)
            shape[0] *= 2
            new_key = np.zeros(tuple(shape))
            new_key[0::2] = key * 2
            new_key[1::2] = new_key[0::2] + 1

            return new_key

    def __getitem__(self, item):
        if self.channels == 2:
            item = self._double_key(item)
            return self.loop_arr.__getitem__(item)
        else:
            return self.loop_arr.__getitem__(item)

    def __setitem__(self, key, value):
        if self.channels == 2:
            if len(value) == 2:
                if isinstance(key, slice):
                    key1 = slice(2 * key.start, 2 * key.stop, 2)
                    key2 = slice(2 * key.start + 1, 2 * key.stop, 2)
                elif isinstance(key, int) or isinstance(key, np.array):
                    key1 = 2 * key
                    key2 = 2 * key + 1
                self.loop_arr.__setitem__(key1, value[0])
                self.loop_arr.__setitem__(key2, value[1])
            else:
                key = self._double_key(key)
                self.loop_arr.__setitem__(key, value)
        else:
            self.loop_arr.__setitem__(key, value)

    def as_buffer(self, range):
        return self[range].tostring()


class LoopArray(object):
    """
    Wraps an array to allow periodic boundary conditions in arbitrary axes when slicing. However, at most one limit can be
    exceeded per axis, i.e. for the index i of array arr, i < 2 * len(arr) must hold.
    Note that this class avoids the construction of index arrays when __getitem__ is called and, if not iterable
    is passed, also when __setitem__ is called.
    """

    def __init__(self, arr, dtype=None, shared=True, modes=None):
        """
        :param arr: np.array
        :param dtype: must be supplied if shared == True
        :param shared: bool, whether the array should be shared memory
        :param modes: list containing modes for each axis ('raise', 'wrap'). If 'wrap', slicing will be periodic.
        """
        if shared:
            if dtype is None:
                raise TypeError('dtype cannot be NoneType when shared == True')
            self.arr = shared_array(arr, dtype)
        else:
            self.arr = arr

        if modes is None:
            modes = []
            for axis in range(len(self.arr.shape) - 1):
                modes.append('raise')
            modes.append('wrap')

        self.modes = modes

    @property
    def shape(self):
        return self.arr.shape

    def _get_indices_slice(self, slic, axis):
        """
        Returns the indices of a slice object.

        :param slic: slice
        :param axis: int
        :return: key_start, key_stop, step
        """
        key_start = slic.start
        key_stop = slic.stop

        if key_start is None:
            key_start = 0
        if key_stop is None:
            key_stop = self.arr.shape[axis]

        if key_stop - key_start > self.arr.shape[axis]:
            raise ValueError('For index i, i < 2 * len(arr) must hold')

        if slic.step is None:
            step = 1
        else:
            step = slic.step

        return key_start, key_stop, step

    def _slice_to_array(self,slic, axis):
        """
        Construct an array of the indices covered by a slice object.

        :param slic: slice
        :param axis: int
        :return: np.array
        """
        return np.arange(*self._get_indices_slice(slic, axis))

    def _modulo_slice(self, slic, axis):
        """
        Wrap start and stop indices of a slice object. Returns a slice object with start and stop indices % arr.shape[axis]
        if slice does not extend beyond the array limits, otherwise a tuple of slices is returned.

        :param slic: slice
        :param axis: int
        :return: slice or (slice, slice)
        """
        key_start, key_stop, step = self._get_indices_slice(slic, axis)
        key_stop -= 1

        start_modulo = key_start % self.arr.shape[axis]
        stop_modulo = key_stop % self.arr.shape[axis]

        if start_modulo > stop_modulo:
            slice1 = slice(start_modulo, self.arr.shape[axis], step)
            step_modulo = (self.arr.shape[axis] - start_modulo) % step
            slice2 = slice(step_modulo, stop_modulo + 1, step)
            return slice1, slice2
        else:
            slice1 = slice(start_modulo, stop_modulo + 1, step)
            return slice1

    def _create_key_for_axis(self, k, axis):
        return [slice(None)] * axis + [k] + [slice(None)] * (len(self.arr.shape) - axis - 1)

    def __getitem__(self, key):
        """
        Get sliced array. Slice can exceed at most one limit per axis.
        :param key: int or slice
        :return: np.array
        """

        if not hasattr(key, '__iter__'):
            key = [key]

        key_ret = []
        for axis, k in enumerate(key):
            if self.modes[axis] == 'wrap':
                if isinstance(k, int):
                    k = k % self.arr.shape[axis]
                    new_key = self._create_key_for_axis(k, axis)
                    key_ret.append(new_key)
                elif isinstance(k, slice):
                    k = self._modulo_slice(k, axis)

                    if isinstance(k, tuple):
                        key1 = self._create_key_for_axis(k[0], axis)
                        key2 = self._create_key_for_axis(k[1], axis)
                        new_key = (key1, key2)
                    else:
                        new_key = self._create_key_for_axis(k, axis)
                    key_ret.append(new_key)
                else:
                    raise TypeError('key must be int or slice, not '+str(type(key))+': '+str(key))
            else:
                new_key = self._create_key_for_axis(k, axis)
                key_ret.append(new_key)

        ret = self.arr
        axis_reduction = 0
        for axis, k in enumerate(key_ret):
            if isinstance(k, tuple):
                ret = np.concatenate((ret[tuple(k[0][axis_reduction:])], ret[tuple(k[1][axis_reduction:])]),
                                     axis=axis - axis_reduction)
            else:
                ret = ret[tuple(k)]
                if isinstance(k, int):
                    axis_reduction += 1

        return ret

    def __setitem__(self, key, val):
        """
        Set slice of array. Slice of array covered by key can exceed at most one limit per axis.

        :param key: int, slice or iterable
        :param val:
        """

        if not hasattr(key, '__iter__'):
            key = [key]

        key = list(key)

        if key.__contains__(Ellipsis):
            i = key.index(Ellipsis)
            nr_pad = len(self.arr.shape) - (len(key) - 1)
            key[i:i+nr_pad] = [slice(None, None, None)] * nr_pad

        ind = []
        for axis, k in enumerate(key):
            if isinstance(k, int):
                app = k
            elif isinstance(k, slice):
                app = self._slice_to_array(k, axis)
            elif hasattr(k, '__iter__'):
                app = np.array(k)
            else:
                raise TypeError('key must be int, slice or have __iter__ attribute')

            if self.modes[axis] == 'wrap':
                app %= self.arr.shape[axis]
            ind.append(app)
        ind = tuple(ind)
        self.arr[ind] = val


class UniqueDict(object):
    """
    A dictionary class with unique keys *and* values.
    """

    def __init__(self):
        self._dict = {}
        self._inv_dict = {}

    def __getitem__(self, key):
        try:
            return self._dict[key]
        except:
            return self._inv_dict[key]

    def __setitem__(self, key, value):
        if value not in self._inv_dict:
            self._dict[key] = value
            self._inv_dict[value] = key
        else:
            raise KeyError('This entry (' + str(value) + ') already exists under key ' + str(self._inv_dict[value]))

    def __delitem__(self, key):
        if key in self._dict:
            value = self._dict[key]
            del self._dict[key]
            del self._inv_dict[value]
        elif key in self._inv_dict:
            value = self._inv_dict[key]
            del self._dict[value]
            del self._inv_dict[key]
        else:
            raise KeyError

    def __iter__(self):
        return self._dict.__iter__()

    def __contains__(self, item):
        return self._dict.__contains__(item) or self._inv_dict.__contains__(item)

    def get_key(self, key):
        if key in self._dict:
            return key
        elif key in self._inv_dict:
            return self._inv_dict[key]
        else:
            raise KeyError(str(key) + ' is not in this UniqueDict')

    def get_inv_key(self, key):
        if key in self._dict:
            return self._dict[key]
        elif key in self._inv_dict:
            return key
        else:
            raise KeyError(str(key) + ' is not in this UniqueDict')

    def change_key(self, key, new_key):
        if key in self._dict:
            tmp = self._dict[key]
            del self._dict[key]
            del self._inv_dict[tmp]

            self._dict[new_key] = tmp
            self._inv_dict[tmp] = new_key

        elif key in self._inv_dict:
            tmp = self._inv_dict[key]
            del self._dict[tmp]
            del self._inv_dict[key]

            self._dict[tmp] = new_key
            self._inv_dict[new_key] = tmp

        else:
            raise KeyError(str(key) + ' is not in this UniqueDict')

    def is_key(self, val):
        return val in self._dict

    def is_inv_key(self, val):
        return val in self._inv_dict

    def __len__(self):
        return len(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()


def shared_array(arr, dtype):
    """
    Comstruct a shared memory numpy array.
    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing
    """

    shape = arr.shape

    np_type_to_ctype = {np.float32: ctypes.c_float,
                        np.float64: ctypes.c_double,
                        np.bool: ctypes.c_bool,
                        np.uint8: ctypes.c_uint8,
                        np.uint64: ctypes.c_ulonglong,
                        np.int: ctypes.c_int8}

    shared_array_base = mp.Array(np_type_to_ctype[dtype], np.multiply.reduce(shape))
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(*shape)

    np.copyto(shared_array, arr.astype(dtype))
    return shared_array


class StoreInterface:
    """
    Interface for classes eligible to be managed by a ui.gtk_utils.Store
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def duplicate(self): raise NotImplementedError

    @abstractmethod
    def get_preview(self, dims): raise NotImplementedError
