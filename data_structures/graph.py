import run_time.config as config
import data_structures.data_utils as data_utils

import bisect
import numpy as np
from scipy.interpolate import splrep, splev
import copy


class Graph(data_utils.StoreInterface):
    """
    Class used to store and manage information about a graph, that is of a mapping xs -> ys.
    Instances can be set is_settable = True if data need to be changed and is_settable = False if the data needs to be
    protected.
    If is_raw_data = False, instances return an interpolation on calling get. The interpolation data is hold and updated on
    change in the data. To enable the interpolation, instances with is_raw_data = False must be set a x-value array with
    set_interpolation_xs.

    To customize the interpolation, insertion rules (rules applied upon calling insert) and filtering (applied upon
    updating the interpolation), override interpolate, apply_insert_point_rules and filter_interpolated.
    """

    def __init__(self, ys, xs, name="graph", raw_data=False, settable=True, on_get_interpolation_unchanged=True):
        self._xs = xs
        self._ys = ys
        self._is_raw_data = raw_data
        self._is_settable = settable
        self.name = name

        self._interp_xs = None
        self._interp_ys = None

        self.on_get_interpolation_unchanged = on_get_interpolation_unchanged
        self.is_changed = True

    @property
    def is_raw_data(self):
        return self._is_raw_data

    @property
    def is_settable(self):
        return self._is_settable

    def apply_insert_point_rules(self, coords):
        """
        By overriding this method, insertion rules can be applied to the graph.

        :param coords:
        :return:
        """
        return True

    def apply_remove_point_rules(self, coords):
        """
        By overriding this method, insertion rules can be applied to the graph.

        :param coords:
        :return:
        """
        return True

    def get_interpolation(self, interpolation_xs=None, force_recalculation=False):
        """
        Gets the interpolation of the current graph in the x values defined either by the kwarg interpolation_xs
        or by the xs set in a previous call of set_interpolate. The interpolation is performed according to the
        implementation in interpolate.

        :param interpolation_xs:
        :param force_recalculation:
        :return: interpolated_ys, interpolated_xs
        """
        interp_xs = self._interp_xs

        if interpolation_xs is not None:
            interp_xs = interpolation_xs
            new_ys = self.interpolate(interp_xs)
            return self.filter_interpolated(new_ys, interp_xs)
        elif interp_xs is None:
            raise ValueError('set_interpolation must be set before if no interpolation_xs is supplied')

        if self.is_changed or self._interp_ys is None or force_recalculation:
            new_ys = self.interpolate(interp_xs)
            if self.on_get_interpolation_unchanged:
                self.is_changed = False
            self._interp_ys, self._interp_xs = self.filter_interpolated(new_ys, interp_xs)
            self.apply_post_interpolate_updates()
            return self._interp_ys, self._interp_xs
        else:
            return self._interp_ys, self._interp_xs

    def interpolate(self, xs):
        """
        Returns an interpolation of the data in the x-values xs.
        :param xs:
        :return: ys
        """
        tck = splrep(self._xs, self._ys)
        new_ys = splev(xs, tck, der=0)
        return new_ys

    def get(self):
        """
        Returns the updated interpolated data if the instance has is_raw_data = False. Else the raw data is returned.

        :return: ys, xs
        """
        if not self.is_raw_data and self._interp_xs is not None:
            return self.get_interpolation()
        return self._ys, self._xs

    @property
    def xs(self):
        if not self.is_raw_data and self._interp_xs is not None:
            ys, xs = self.get_interpolation()
            return xs
        return self._xs

    @property
    def ys(self):
        if not self.is_raw_data and self._interp_xs is not None:
            ys, xs = self.get_interpolation()
            return ys
        return self._ys


    def get_raw_graph(self):
        """
        :return: ys, xs
        """
        return self._ys, self._xs

    def get_raw_ys(self):
        return self._ys

    def get_raw_xs(self):
        return self._xs

    def _insert_xs(self, ind, val):
        self._xs = np.insert(self._xs, ind, val)

    def _insert_ys(self, ind, val):
        self._ys = np.insert(self._ys, ind, val)

    def _remove_xs(self, ind):
        self._xs = np.delete(self._xs, ind)

    def _remove_ys(self, ind):
        self._ys = np.delete(self._ys, ind)

    def insert(self, y, x):
        """
        Insert a data point if it satisfies the conditions specified in apply_insert_point_rules. Insertion is only possible
        for instances that are is_settable.
        :param y:
        :param x:
        :return:
        """
        assert self.apply_insert_point_rules((y, x)), 'Insertion rules are not satisfied'

        if self.is_settable:
            i = bisect.bisect_left(self._xs, x)
            self._insert_xs(i, x)
            self._insert_ys(i, y)
            self.is_changed = True
            return i
        else:
            raise ValueError('graph '+str(self.name)+' is not settable')

    def remove(self, i):
        """
        Remove data point with index i.

        :param i:
        :return:
        """
        assert self.apply_remove_point_rules((self._ys[i], self._xs[i])), 'Removal rules are not satisfied'

        if len(self.get_raw_xs()) > 5:
            if self.is_settable:
                self._remove_xs(i)
                self._remove_ys(i)
                self.is_changed = True
            else:
                raise ValueError('graph '+str(self.name)+' is not is_settable')
        elif not self.is_raw_data:
            raise ValueError('Must be at least 5 points for interpolation.')

    def set_interpolation_xs(self, vals):
        """
        Set the array of x-values for which the interpolation is returned.

        :param vals:
        :return:
        """
        vals = np.array(vals)
        self._interp_xs = vals
        self.is_changed =True
        # self._interp_ys = None
        self.get_interpolation()

    def filter_interpolated(self, ys, xs):
        """
        Filter the interpolated values.

        :param ys:
        :param xs:
        :return: filtered_xs, filtered_ys
        """
        return ys, xs

    def apply_post_interpolate_updates(self):
        pass

    def get_preview(self, dims):
        pass

    def duplicate(self):
        return copy.deepcopy(self)


class Limiter(Graph):
    """
    A graph with x-values ranging from 0 to 1 with raw_data = False. The interpolation has a lower bound at 0.
    """
    def __init__(self, length, *args, **kwargs):
        # gtk_utils.GtkListener.__init__(self)
        Graph.__init__(self, np.ones(length), np.linspace(0, 1, length), *args, **kwargs)

        self._width = config.params.image_width
        self.set_interpolation_xs(self.interpolation_range)

        self._envelope = data_utils.LoopArray(self.ys, self.ys.dtype, shared=False, modes=['wrap'])

        self.name = 'Limiter'
        # self.listen_to('name')

    @property
    def envelope(self):
        return self._envelope

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, val):
        self._width = val
        self.set_interpolation_xs(self.interpolation_range)

    @property
    def interpolation_range(self):
        return np.linspace(0, 1, config.params.image_width)

    def filter_interpolated(self, ys, xs):
        return np.clip(ys, 0, None), xs

    def apply_post_interpolate_updates(self):
        self._envelope = data_utils.LoopArray(self._interp_ys, self._interp_ys.dtype, shared=False, modes=['wrap'])


class Fader(Graph):
    def __init__(self, length, fade_in=True, *args, **kwargs):
        if fade_in:
            ys = np.linspace(0, 1, length)
        else:
            ys = np.linspace(1, 0, length)

        Graph.__init__(self, ys, np.linspace(0, 1, length), *args, **kwargs)

        self._width = config.project.params.sound_gen_params.transition_length
        self.set_interpolation_xs(self.interpolation_range)

        self.name = 'Fader'
        # self.listen_to('name')

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, val):
        val = min(val, config.project.params.sound_gen_params.samples_per_px)
        self._width = val
        self.set_interpolation_xs(self.interpolation_range)

    @property
    def interpolation_range(self):
        return np.linspace(0, 1, self.width)


class FrequencyMap(Graph):
    """
    A map from px in y direction to frequency . Traditional as well as arbitrary tonal systems can be implemented by
    FrequencyMap.

    Default (fixed) maps are best implemented by deriving from FrequencyMap, defining first the map in the constructor,
    then calling the super __init__.
    """
    def __init__(self, base=None, ys=None, xs=None, *args, **kwargs):

        # if base is supplied, copy everything from base,
        # else run constructor
        if isinstance(base, self.__class__):
            self.__dict__ = base.__dict__
        elif ys is not None and xs is not None:
            Graph.__init__(self, ys, xs, raw_data=True, settable=True, on_get_interpolation_unchanged=False,
                                 *args, **kwargs)
        else:
            raise ValueError("Either kwarg base or kwargs ys and xs must be defined.")

    def get_map(self):
        """
        Get range of map between start and end px. This method allows to store abstract frequency maps without information
        that is specific to the track.

        :param start:
        :param end:
        :return:
        """
        return self.get_raw_ys()


class LinearFrequencyMap(FrequencyMap):
    def __init__(self, length):
        ys = np.linspace(200, 400, length)
        xs = ys
        FrequencyMap.__init__(self, ys=ys, xs=xs, name='LinearFrequencyMap')