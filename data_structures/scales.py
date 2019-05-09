import numpy as np
import data_structures.graph as graph
import run_time.config as config
import ui.base_widgets as base_widgets
import re

from abc import ABCMeta, abstractmethod

class StandardScaleToMap(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get(self):
        raise NotImplementedError

    @abstractmethod
    def set(self):
        raise NotImplementedError

    @abstractmethod
    def settings_table(self):
        raise NotImplementedError


class LinearFrequencyMap(StandardScaleToMap):
    def __init__(self, length=None):
        self.length = length
        self.ys = None
        self.xs = None

    def get(self):
        return graph.FrequencyMap(ys=self.ys, xs=self.xs, name='LinearFrequencyMap')

    def set(self):
        self.ys = np.linspace(200, 400, self.length)
        self.xs = self.ys
        return self

    def settings_table(self):
        settings = [('Length', 'length', int, 'entry', {}, 1, 1)]
        return base_widgets.SettingsGrid(settings, self)


class OctaveBasedStandardScale(StandardScaleToMap):
    def __init__(self):
        self.octaves = [4]
        self.ys = None
        self.xs = None

    def get(self):
        return graph.FrequencyMap(ys=np.array(self.ys), xs=self.xs, name='StandardScale')

    def _digit_str_to_np(self, s):
        s = ' '.join(c for c in s if c.isdigit())
        return np.fromstring(s, sep=' ')

    def settings_table(self):
        settings = [('Octaves', 'octaves', str, 'entry', {}, 1, 1)]
        return base_widgets.SettingsGrid(settings, self)


class PythagoreanScaleMap(OctaveBasedStandardScale):
    def set(self):
        whole_tone = 9 / 8
        semi_tone = 256 / 243
        if type(self.octaves) == str:
            self.octaves = self._digit_str_to_np(self.octaves)
        ys = []
        xs = np.arange(0, len(self.octaves) * 7, 1)
        for oct in self.octaves:
            a = config.params.concert_a * 2 ** (oct - 4)
            b = a * whole_tone
            g = a / whole_tone
            f = g / whole_tone
            e = f / semi_tone
            d = e / whole_tone
            c = d / whole_tone
            ys += [c, d, e, f, g, a, b]

        self.ys, self.xs = np.array(ys), xs
        return self


class DiatonicScaleMap(OctaveBasedStandardScale):
    def set(self):
        major_tone = 9 / 8
        minor_tone = 10 / 9
        semi_tone = 16 / 15
        if type(self.octaves) == str:
            self.octaves = self._digit_str_to_np(self.octaves)
        ys = []
        xs = np.arange(0, len(self.octaves) * 7, 1)
        for oct in self.octaves:
            a = config.params.concert_a * 2 ** (oct - 4)
            b = a * major_tone
            g = a / minor_tone
            f = g / major_tone
            e = f / semi_tone
            d = e / minor_tone
            c = d / major_tone
            ys += [c, d, e, f, g, a, b]

        self.ys, self.xs = np.array(ys), xs
        return self


class ChromaticScaleMap(OctaveBasedStandardScale):
    def set(self):
        major_tone = 9 / 8
        minor_tone = 10 / 9
        semi_tone = 16 / 15
        interval = 25 / 24
        if type(self.octaves) == str:
            self.octaves = self._digit_str_to_np(self.octaves)
        ys = []
        for oct in self.octaves:
            a = config.params.concert_a * 2 ** (oct - 4)
            b = a * major_tone
            g = a / minor_tone
            f = g / major_tone
            e = f / semi_tone
            d = e / minor_tone
            c = d / major_tone

            scale = [c, d, e, f, g, a, b]
            for freq in scale:
                flat, sharp = freq / interval, freq * interval
                ys += [flat, freq, sharp]

        self.ys, self.xs = np.array(ys), np.arange(0, len(ys))
        return self


class TemperedScaleMap(OctaveBasedStandardScale):
    def set(self):
        semi_tone = 2 ** (1 / 12)
        if type(self.octaves) == str:
            self.octaves = self._digit_str_to_np(self.octaves)
        ys = []
        for oct in self.octaves:
            a = config.params.concert_a * 2 ** (oct - 4)
            a_sharp = a * semi_tone
            b = a_sharp * semi_tone
            g_sharp = a / semi_tone
            g = g_sharp / semi_tone
            f_sharp = g / semi_tone
            f = f_sharp / semi_tone
            e = f / semi_tone
            d_sharp = e / semi_tone
            d = d_sharp / semi_tone
            c_sharp = d / semi_tone
            c = c_sharp / semi_tone

            scale = [c, c_sharp,
                     d, d_sharp,
                     e, f,
                     f_sharp, g,
                     g_sharp, a,
                     a_sharp, b]
            ys += scale

        self.ys, self.xs = np.array(ys), np.arange(0, len(ys))
        return self

