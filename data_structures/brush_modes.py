import data_structures.data_utils as data_utils
import ui.base_widgets as base_widgets
import run_time.config as config
import run_time.py_utils as py_utils

import numpy as np

from abc import ABCMeta, abstractmethod


class ApplyMode(data_utils.StoreInterface):
    """
    Interface for Brush Modes.

    A Brush Mode implements the logic the apply method of Brush.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply(self, arr1, arr2, mask, weights):
        raise NotImplementedError

    def duplicate(self):
        raise Exception('BrushMode instances cannot be duplicated')

    def get_preview(self, dims):
        raise NotImplementedError

    def show_settings_dialog(self):
        return None


class BrushMode(ApplyMode):
    __metaclass__ = ABCMeta

    def __init__(self):
        ApplyMode.__init__(self)

        self.upper_thr = 255
        self.lower_thr = 0
        self.include_empty = True

        settings = [('Upper Threshold', 'upper_thr', int, 'entry', {'range': (0, 255), 'increments': (1, 1)}, 1, 1),
                    ('Lower Threshold', 'lower_thr', int, 'entry', {'range': (0, 255), 'increments': (1, 1)}, 1, 1),
                    ('Include Empty', 'include_empty', bool, 'switch', {}, 1, 1)]
        self.settings_dialog = base_widgets.SettingsDialog(settings, self, 'Set Threshold', config.main_window)



class SubtractMode(BrushMode):
    """ Subtract without reweighing canvas image."""

    def apply(self, img, brush_img, mask, weights):
        """
        Return weighted difference of two arrays.

        :param mask:
        :param img: np.array
        :param brush_img: np.array
        :param weights: np.array, opacity of brush_img
        :return:
        """
        brush_img = brush_img.astype('float')

        shape = (weights.shape[0], weights.shape[1], img.shape[2])
        bg = np.reshape(img[mask].astype('float'), shape)

        s = np.sum(bg, axis=2) / 3
        thr_mask = np.logical_and.reduce((s <= self.upper_thr,
                                          s >= self.lower_thr,
                                          np.sum(brush_img, axis=2) > self.include_empty))
        not_thr_mask = np.logical_not(thr_mask)

        br = np.einsum('ijk, ij -> ijk', brush_img, weights)
        # bg = np.einsum('ijk, ij -> ijk', np.reshape(img[mask].astype('float'), shape), 1 / (1 - weights))

        bg[thr_mask] -= br[thr_mask]
        bg[not_thr_mask] = bg[not_thr_mask]

        img[mask] = np.ravel(np.clip(bg, a_min=0, a_max=255).astype('uint8'))
        return img

    def show_settings_dialog(self):
        self.settings_dialog.run()


class NegativeMixingMode(BrushMode):
    """ Subtract from reweighed canvas image. """

    def apply(self, img, brush_img, mask, weights):
        """
        Return weighted difference of two arrays.

        :param mask:
        :param img: np.array
        :param brush_img: np.array
        :param weights: np.array, opacity of brush_img
        :return:
        """
        brush_img = brush_img.astype('float')

        shape = (weights.shape[0], weights.shape[1], img.shape[2])
        bg = np.reshape(img[mask].astype('float'), shape)

        s = np.sum(bg, axis=2) / 3
        thr_mask = np.logical_and.reduce((s <= self.upper_thr,
                                          s >= self.lower_thr,
                                          np.sum(brush_img, axis=2) > self.include_empty))
        not_thr_mask = np.logical_not(thr_mask)

        br = np.einsum('ijk, ij -> ijk', brush_img, weights / (1 - weights))
        bg = np.einsum('ijk, ij -> ijk', bg, 1 / (1 - weights))

        bg[thr_mask] -= br[thr_mask]
        bg[not_thr_mask] = bg[not_thr_mask]

        img[mask] = np.ravel(np.clip(bg, a_min=0, a_max=255).astype('uint8'))
        return img

    def show_settings_dialog(self):
        self.settings_dialog.run()


class AdditiveMixingMode(BrushMode):
    """ Add to reweighed canvas image. """

    def apply(self, img, brush_img, mask, weights):
        """
        Return sum of two arrays. Weights holds the opacity of brush_img

        :param mask:
        :param img: np.array
        :param brush_img: np.array
        :param weights: np.array, opacity of brush_img
        :return:
        """
        brush_img = brush_img.astype('float')

        shape = (weights.shape[0], weights.shape[1], img.shape[2])
        bg = np.reshape(img[mask].astype('float'), shape)

        s = np.sum(bg, axis=2) / 3
        thr_mask = np.logical_and.reduce((s <= self.upper_thr,
                                          s >= self.lower_thr,
                                          np.sum(brush_img, axis=2) > self.include_empty))
        not_thr_mask = np.logical_not(thr_mask)
        bg = np.einsum('ijk, ij -> ijk', bg, (1 - weights))
        br = np.einsum('ijk, ij -> ijk', brush_img, weights)

        bg[thr_mask] += br[thr_mask]
        bg[not_thr_mask] = bg[not_thr_mask]

        img[mask] = np.ravel(bg).astype('uint8')

        return img

    def show_settings_dialog(self):
        self.settings_dialog.run()


class AddMode(BrushMode):
    """ Add without reweighing canvas image. """

    def apply(self, img, brush_img, mask, weights):
        """
        Return weighted difference of two arrays.

        :param mask:
        :param img: np.array
        :param brush_img: np.array
        :param weights: np.array, opacity of brush_img
        :return:
        """
        brush_img = brush_img.astype('float')

        shape = (weights.shape[0], weights.shape[1], img.shape[2])
        bg = np.reshape(img[mask].astype('float'), shape)

        s = np.sum(bg, axis=2) / 3

        thr_mask = np.logical_and.reduce((s <= self.upper_thr,
                                          s >= self.lower_thr,
                                          np.sum(brush_img, axis=2) > self.include_empty))
        not_thr_mask = np.logical_not(thr_mask)

        br = np.einsum('ijk, ij -> ijk', brush_img, weights)

        bg[thr_mask] += br[thr_mask]
        bg[not_thr_mask] = bg[not_thr_mask]

        img[mask] = np.ravel(np.clip(bg, a_min=0, a_max=255).astype('uint8'))
        return img

    def show_settings_dialog(self):
        self.settings_dialog.run()


class FilterMode(ApplyMode):
    """ Apply a PIL filter to a selected patch. """

    def __init__(self):
        self.filter_type = 'gaussian_blur'
        self.size = 0
        self.axis= 0
        self.upper_thr = 255
        self.lower_thr = 0
        self.include_empty = True

        settings = [('Filter Type', 'filter_type', str, 'combo',
                     {'store': py_utils.ImageHandler.get_support_keys()}, 1, 1),
                    ('Size', 'size', int, 'entry', {}, 1, 1),
                    ('Axis', 'axis', int, 'entry', {'range': (0, 1)}, 1, 1),
                    ('Upper Threshold', 'upper_thr', int, 'entry', {'range': (0, 255), 'increments': (1, 1)}, 1, 1),
                    ('Lower Threshold', 'lower_thr', int, 'entry', {'range': (0, 255), 'increments': (1, 1)}, 1, 1),
                    ('Include Empty', 'include_empty', bool, 'switch', {}, 1, 1)]

        self.settings_dialog = base_widgets.SettingsDialog(settings, self, 'Set Threshold', config.main_window)

    def apply(self, img, brush_img, mask, weights):
        """
        Return weighted difference of two arrays.

        :param mask:
        :param img: np.array
        :param brush_img: np.array
        :param weights: np.array, opacity of brush_img
        :return:
        """

        shape = (weights.shape[0], weights.shape[1], img.shape[2])
        bg = np.reshape(img[mask], shape).astype('float64')

        br = py_utils.ImageHandler.Filter(self.filter_type, bg.astype('uint8'),
                                                 self.size, self.axis).reshape(shape).astype('float64')

        s = np.sum(bg, axis=2) / 3
        thr_mask = np.logical_and.reduce((s <= self.upper_thr,
                                          s >= self.lower_thr,
                                          np.sum(br, axis=2) > self.include_empty))
        not_thr_mask = np.logical_not(thr_mask)

        bg = np.einsum('ijk, ij -> ijk', bg, (1 - weights))
        br = np.einsum('ijk, ij -> ijk', br, weights)

        bg[thr_mask] += br[thr_mask]
        bg[not_thr_mask] = bg[not_thr_mask]

        img[mask] = np.ravel(np.clip(bg, a_min=0, a_max=255).astype('uint8'))
        return img

    def show_settings_dialog(self):
        self.settings_dialog.run()