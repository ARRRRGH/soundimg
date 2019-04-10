import run_time.py_utils as py_utils
import run_time.config as config
import data_structures.data_utils as data_utils

import copy
import numpy as np

from abc import ABCMeta, abstractmethod


class SingleColorBrush(object):
    """
    Brushes of a single color should derive from this class. The abstract method set_color is used by menus implementing
    a color choice of the user.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def set_color(self, color): raise NotImplementedError


class Brush(data_utils.StoreInterface):
    """
    Any brush must derive from this abstract class. A brush consists in two arrays representing the colors and weights
    of a certain shape. These two arrays are used for the application of the brush to an image by a function defined by
    mode.
    The shape is either fixed or dynamically set depending on the image's shape the brush acts on and the axis mode. See
    set_axis_modes for the currently implemented axis modes.
    Colors and weights are updated upon calling the abstract method update. Child classes define their own update as
    they might use additional structures that influence the update.
    """
    __metaclass__ = ABCMeta

    def __init__(self, mode, *args, **kwargs):
        self._current_shape = (1, 1)

        self.mode = mode
        self.weights = None
        self.colors = None

        self.axis_modes = (None, None)
        self._axis_mode_dict = {'extend': self._shape_extend, 'single_px': self._shape_single_px}

    @property
    def shape(self):
        return self._current_shape

    def set_mode(self, mode):
        self.mode = mode

    def _get_shape_from_image(self, imag, *args, **kwargs):
        """
        Get the shape of the brush given imag. Its dependency from imag allows the implementation of child
        classes that dynamically adapt their shape to the images they are applied to.

        :param imag: image.Image object
        :param args:
        :param kwargs:
        :return:
        """
        s = list(self._current_shape)
        if s[0] is None:
            s[0] = self.get_shape_in_axis(0, imag, *args, **kwargs)
        if s[1] is None:
            s[1] = self.get_shape_in_axis(1, imag, *args, **kwargs)
        return tuple(s)

    def add_axis_mode(self, name, func):
        self._axis_mode_dict[name] = func

    def set_axis_modes(self, axis_modes):
        """
        Set the axis modes. Currently, these modes are implemented
            'extend': extend brush over the whole image
            'single_px': fix shape in one direction to one px

        :param axis_modes: 2-tuple of str for both directions
        :return:
        """
        if axis_modes[0] is not None:
            assert axis_modes[0] in self._axis_mode_dict, 'use first add_axis_mode to add ' + str(axis_modes[0])
        if axis_modes[1] is not None:
            assert axis_modes[1] in self._axis_mode_dict, 'use first add_axis_mode to add '  + str(axis_modes[1])

        self.axis_modes = axis_modes
        self._current_shape = tuple([None if isinstance(axis_mode, str)
                                     else self._current_shape[i] for i, axis_mode in enumerate(axis_modes)])

    def __getstate__(self):
        """
        For pickling.

        :return:
        """
        return self._current_shape, self.mode, self.weights, self.colors, self.axis_modes

    def __setstate__(self, state):
        """
        For pickling.

        :param state:
        :return:
        """
        self._current_shape, self.mode, self.weights, self.colors, self.axis_modes = state

    def set_shape_from_scalar(self, scalar):
        """
        Prepare the scaling of the brush according to its axis modes upon update.

        :param scalar:
        :return:
        """
        if scalar != 0:
            s = self.shape
            if s[0] is None and s[1] is None:
                pass
            elif s[0] is None:
                self._current_shape = (None, int(scalar))
            elif s[1] is None:
                self._current_shape = (int(scalar), None)
            else:
                frac = scalar / float(max(s[0], s[1]))
                self._current_shape = (int(s[0] * frac), int(s[1] * frac))

    def get_shape_in_axis(self, axis, imag, *args, **kwargs):
        """
        Convenience function to get the shape in one axis according to the corresponding axis mode.

        :param axis:
        :param imag:
        :param args:
        :param kwargs:
        :return: int
        """
        return self._axis_mode_dict[self.axis_modes[axis]](axis, imag, *args, **kwargs)

    @staticmethod
    def _shape_single_px(axis, imag, *args, **kwargs):
        return 1

    @staticmethod
    def _shape_extend(axis, imag, *args, **kwargs):
        return 2 * imag.img.shape[axis]

    def apply(self, event, imag):
        """
        Apply the brush to imag on the area centered on event. Returns True or False depending on the successful outcome
        of the application.

        :param event: 2-tuple of int or matplotlib event, center coordinates
        :param imag:
        :return:
        """
        self.update(imag)
        try:
            mask, weights, colors = self.get_framed_brush(event, imag)
        except ValueError:
            return False

        changed_img = self.mode.apply(imag.img, colors, mask, weights)
        imag.set_img(mask, changed_img)
        return True

    @staticmethod
    def unpack_event(event):
        """
        Get right event tuple form.

        :param event:
        :return: y, x
        """
        if hasattr(event, '__iter__'):
            x = event[1]
            y = event[0]
        else:
            x = int(event.xdata)
            y = int(event.ydata)

        return y, x

    def get_bounds(self, event, imag):
        """
        Get bounds of the brush on the image imag.

        :param event:
        :param imag: image.Image
        :return: xmin, xmax, ymin, ymax
        """
        y, x = self.unpack_event(event)

        img = imag.img
        s = self._get_shape_from_image(imag)

        xmin = self._set_in_boundaries(x - np.floor(s[1] / 2.), img.shape[1])
        xmax = self._set_in_boundaries(x + np.ceil(s[1] / 2.), img.shape[1])
        ymin = self._set_in_boundaries(y - np.floor(s[0] / 2.), img.shape[0])
        ymax = self._set_in_boundaries(y + np.ceil(s[0] / 2.), img.shape[0])

        assert not (xmin == xmax or ymin == ymax), 'xmin == xmax or ymin == ymax'

        return xmin, xmax, ymin, ymax

    def get_framed_brush(self, event, imag):
        """
        Get a mask defining the area on which the brush acts, as well the weights and colors, cut on the edges if the
        brush overlaps the image limits.

        :param event:
        :param imag:
        :return: mask, weight, colors
        """
        xmin, xmax, ymin, ymax = self.get_bounds(event, imag)

        grid = imag.grid
        masked_array = np.ma.masked_where(np.logical_and.reduce((grid[0] >= ymin,
                                                                 grid[0] < ymax,
                                                                 grid[1] >= xmin,
                                                                 grid[1] < xmax)), imag.img)
        mask = masked_array.mask
        weights = self.weights
        colors = self.colors
        weights, colors = self.cut(imag.shape, weights, colors, ymin, ymax, xmin, xmax)

        return mask, weights, colors

    @staticmethod
    def _set_in_boundaries(index, length):
        return np.int(min(max(index, 0), length))

    @staticmethod
    def cut(shape, weights, colors, ymin, ymax, xmin, xmax):
        """
        Cut weights and colors if the brush's bounds are over the limits given by shape.

        :param shape:
        :param weights:
        :param colors:
        :param ymin:
        :param ymax:
        :param xmin:
        :param xmax:
        :return: weights, colors
        """
        xlen = xmax - xmin
        ylen = ymax - ymin
        if xmin == 0:
            weights = weights[:, -xlen:]
            colors = colors[:, -xlen:, :]
        if xmax == shape[1]:
            weights = weights[:, :xlen]
            colors = colors[:, :xlen, :]
        if ymin == 0:
            weights = weights[-ylen:, :]
            colors = colors[-ylen:, :, :]
        if ymax == shape[0]:
            weights = weights[:ylen, :]
            colors = colors[:ylen, :, :]
        return weights, colors

    def reset(self, weights, colors, axis_modes):
        self.weights = weights
        self.colors = colors
        self.set_axis_modes(axis_modes)

    def set_weights(self, weights):
        self.weights = weights

    def set_color(self, color):
        self.colors = color

    def duplicate(self):
        """
        For StorageInterface
        :return:
        """
        return copy.deepcopy(self)

    @abstractmethod
    def update(self, imag):
        """
        Must implement the update of weights and colors depending on the brush's shape and the image it acts on.
        This method is called before a brush is applied to an image imag.
        :param imag:
        :return:
        """
        raise NotImplementedError


class PaintedBrush(Brush):
    def __init__(self, orig_weights, orig_colors, *args, **kwargs):
        Brush.__init__(self, *args, **kwargs)
        self.orig_weights = orig_weights
        self.orig_colors = orig_colors
        self.orig_shape = orig_weights.shape

        self.weights = orig_weights.copy()
        self.colors = orig_colors.copy()
        self.weight = 1

        self.editable = True

    @property
    def shape(self):
        return self.orig_weights.shape

    @shape.setter
    def shape(self, val):
        self._current_shape = val

    def update(self, imag, shape=None, ret=False):
        s = self._get_shape_from_image(imag)
        self.weights = py_utils.resize(self.orig_weights * self.weight, (s[1], s[0]))
        self.colors = py_utils.resize(self.orig_colors, (s[1], s[0]))

    def get_preview(self, dims):
        from data_structures import imaging
        updated = imaging.UpdatedImage(img=self.orig_colors, dims_canvas=dims)
        updated.remove_toolbar()
        return updated

    def set_weights(self, weight):
        self.weight = weight


    def __getstate__(self):
        return self._current_shape, self.mode, self.orig_weights, self.orig_colors, self.axis_modes

    def __setstate__(self, state):
        self._current_shape, self.mode, self.orig_weights, self.orig_colors, self.axis_modes = state


class EditorBrush(PaintedBrush):
    def __init__(self, img_updater, orig_weights, *args, **kwargs):
        self.img_updater = img_updater
        self.orig_colors = self.img_updater()
        PaintedBrush.__init__(self, orig_weights, self.orig_colors,
                              mode=config.brush_mode_store['Additive Mixing'],*args, **kwargs)

    def update(self, *args, **kwargs):
        self.orig_colors = self.img_updater()
        super(EditorBrush, self).update(*args, **kwargs)

    def duplicate(self):
        self.orig_colors = self.img_updater()
        return PaintedBrush(self.orig_weights.copy(), self.orig_colors.copy())


class Rectangle(Brush, SingleColorBrush):
    """
    A rectangular, single-colored brush
    """
    def __init__(self, *args, **kwargs):
        Brush.__init__(self, *args, **kwargs)
        self.editable = False
        self.color = np.array([100, 100, 100])
        self.weight = 1

    def update(self, imag):
        s = self._get_shape_from_image(imag)
        self.weights = np.ones(s) * self.weight
        ones = np.ones(s)
        self.colors = np.einsum('ij, k -> ijk', ones, self.color)

    def get_preview(self, dims):
        from data_structures import imaging
        imag_dims = config.dims_preview_image
        updated_image = imaging.UpdatedImage(dims=imag_dims, dims_canvas=dims)
        center = (updated_image.imag.shape[0] // 2, updated_image.imag.shape[1] // 2)
        self.set_shape_from_scalar(int(imag_dims[0] * 0.5))
        self.apply(center, updated_image.imag)
        updated_image.refresh()

        return updated_image

    def set_color(self, color):
        self.color = color

    def set_weights(self, weights):
        self.weight = weights


class InfiniteLine(Rectangle):
    def __init__(self, *args, **kwargs):
        Rectangle.__init__(self, *args, **kwargs)
        axis_modes = ('single_px', 'extend')
        self.set_axis_modes(axis_modes)


class Line(Rectangle):
    def __init__(self, *args, **kwargs):
        Rectangle.__init__(self, *args, **kwargs)
        axis_modes = ('single_px', None)
        self.set_axis_modes(axis_modes)


class Repeater(py_utils.ObjectWrapper):
    """
    A Repeater object wraps a brush object and applies it multiple times to the image.
    """
    def __init__(self, repeated_brush, spacing, nr_repetitions, x_axis_repeated=True, y_axis_repeated=False):
        py_utils.ObjectWrapper.__init__(self, repeated_brush)

        self.repeated_brush = repeated_brush

        self.spacing = spacing
        self.nr_repetitions = nr_repetitions

        self.y_axis_repeated = y_axis_repeated
        self.x_axis_repeated = x_axis_repeated

    def apply(self, event, imag):
        y_init, x_init = self.unpack_event(event)
        self.repeated_brush.update(imag)

        if self.y_axis_repeated:
            event = [y_init, x_init]
            for i in range(self.nr_repetitions):
                event[0] = y_init + i * self.spacing + i * self.weights.shape[0]
                self.repeated_brush.apply(event, imag)

        if self.x_axis_repeated:
            event = [y_init, x_init]
            for i in range(self.nr_repetitions):
                event[1] = x_init + i * self.spacing + i * self.weights.shape[1]
                self.repeated_brush.apply(event, imag)
