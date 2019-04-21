import numpy as np
from numba import jit
import scipy.signal
import data_structures.graph as graph

N = 10000


class WaveTable(graph.Graph):
    @jit(fastmath=True)
    def get_ind(self, ind):
        ind %= self.length
        return np.take(self.get_raw_ys(), ind.astype(np.int64))

    def get_phases(self, l):
        phas = np.linspace(0, 2 * np.pi, N)
        self.length = len(phas)

        return phas

    def show_settings_dialog(self):
        return None


class Sine(WaveTable):
    def __init__(self):

        self.length = N

        phas = self.get_phases(N)
        wav = np.sin(phas)

        WaveTable.__init__(self, wav, phas, name='Sine', raw_data=True, settable=False)


class WhiteNoise(WaveTable):
    def __init__(self):

        self.length = N

        phas = self.get_phases(N)
        wav = np.random.uniform(0, 0.2, N)

        WaveTable.__init__(self, wav, phas, name='Noise', raw_data=True, settable=False)


class SawTooth(WaveTable):
    def __init__(self):

        self.length = N

        phas = self.get_phases(N)
        wav = scipy.signal.sawtooth(phas)

        WaveTable.__init__(self, wav, phas, name='SawTooth', raw_data=True, settable=False)


class Square(WaveTable):
    def __init__(self):

        self.length = N

        phas = self.get_phases(N)
        wav = scipy.signal.square(phas)

        WaveTable.__init__(self, wav, phas, name='Square', raw_data=True, settable=False)


class Ricker(WaveTable):
    def __init__(self):

        self.length = N

        phas = self.get_phases(N)
        wav = scipy.signal.ricker(N, N/40)

        WaveTable.__init__(self, wav, phas, name='Ricker', raw_data=True, settable=False)
