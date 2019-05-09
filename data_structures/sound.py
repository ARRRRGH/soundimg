import data_structures.scales
import run_time.config as config
import synthesis.synth as synth
import data_structures.graph as graph
import data_structures.data_utils as data_utils

import numpy as np
import copy
import scipy.signal


class Instrument(data_utils.StoreInterface):
    """ Instruments hold and update the data and specifications necessary for phase translation in synth.

    In the moment, Instrument holds a wave table, a timbre object and faders. Other objects might be contained in the
    future."""

    def __init__(self, wtab):
        # gtk_utils.GtkListener.__init__(self)

        self.timbre = Timbre()
        self.nr_osc_params = 8
        self.osc_params = self.get_osc_params_from_timbre()

        self.wave_table = copy.deepcopy(wtab)

        self.fader_in = graph.Fader(10, fade_in=True)
        self.fader_out = graph.Fader(10, fade_in=False)

        self._fade_arrs = None

        self.constituents = []
        self.weights = {}
        self.freqs = {}
        self.shifts = {}
        self.wav_table_options = {'constituents': self.constituents,
                                  'weights': self.weights,
                                  'freqs': self.freqs,
                                  'shifts': self.shifts}

        self.name = 'Basic Sine'
        # self.listen_to('name')

        self.update()

    @property
    def fade_arrs(self):
        return self._fade_arrs

    @property
    def nr_of_part_waves(self):
        return self.timbre.shape[0]

    def get_osc_params_from_wav_table_opts(self, opts, constituent):
        osc_params = np.zeros((1, self.nr_osc_params))
        osc_params[:, 4] = opts['freqs'][constituent]
        return osc_params
    
    def get_osc_params_from_timbre(self):
        timbre = self.timbre.partial_waves
        osc_params = np.zeros((timbre.shape[0], self.nr_osc_params), dtype=config.params.dtype)

        norm = np.sum(timbre[:, 0])
        osc_params[:, 0] = timbre[:, 0]  # weight
        osc_params[:, 1] = timbre[:, 0] * timbre[:, 1] / 2. / norm  # a
        osc_params[:, 2] = timbre[:, 0] * (1. - timbre[:, 1]) / norm  # b
        osc_params[:, 3] = timbre[:, 2]  # amf
        osc_params[:, 4] = timbre[:, 3]  # harmonic
        osc_params[:, 5] = timbre[:, 4]  # fma
        osc_params[:, 6] = timbre[:, 5]  # fmf
        osc_params[:, 7] = timbre[:, 6]  # shift

        return osc_params
    
    def update_wav_table(self):
        if len(self.constituents) > 0:
            opts = self.wav_table_options

            # fixme: assumes all wavetables have same length
            length = self.constituents[0].length
            fundamental_freq = synth.get_fundamental_freq(self.constituents[0], config.params.output_sample_rate)

            ones = np.ones(length, dtype=np.float64)
            vals = np.zeros(length, dtype=np.float64)
            norm = 1e-7
            outer_phases = np.linspace(0, 2 * np.pi, length)

            for constituent in self.constituents:
                osc_params = self.get_osc_params_from_wav_table_opts(opts, constituent)
                freqs = ones * fundamental_freq

                inner_phases = synth._get_inner_phases(freqs, outer_phases, osc_params, fundamental_freq)

                # slice due to shape of _get_inner_phases
                wav = (constituent.get_ind(inner_phases) * opts['weights'][constituent])[:, 0]

                norm_per_constituent = opts['weights'][constituent]
                norm += norm_per_constituent
                wav *= scipy.signal.gausspulse(outer_phases - np.pi, fc=opts['shifts'][constituent])
                vals += wav

        # fixme: dont access private graph vars directly
            self.wave_table._ys = vals / norm

    def update_osc_params(self, modulation_options):
        self.timbre.update(modulation_options)
        self.osc_params = self.get_osc_params_from_timbre()

    def update_fade_arrs(self):
        self._fade_arrs = self.fader_in.get()[0], self.fader_out.get()[0]

    def update(self):
        self.update_fade_arrs()
        self.update_wav_table()

    def get_preview(self, dims):
        return self.wave_table.get_preview(dims)

    def remove_constituent(self, constituent):
        for opt_name, opt in self.wav_table_options.items():
            if opt_name == 'constituents':
                opt.remove(constituent)
            else:
                del opt[constituent]

    def add_constituent(self, constituent, weight, freq, shift):
        self.constituents.append(constituent)
        self.weights[constituent] = weight
        self.freqs[constituent] = freq
        self.shifts[constituent] = shift

    def duplicate(self):
        return copy.deepcopy(self)


class FrequencyMapTuple(object):
    """ FrequencyMapTuple assembles an arbitrary number of FrequencyMaps to a new FrequencyMap connecting them
    by a specified transition.

    FrequencyMapTuple stores the last returned frequency map, such that it must only be recalculated when either
    start_map or end_map change.
    """

    def __init__(self, start_map, end_map=None, transition=None):
        """

        :param start_map:
        :param end_map:
        :param transition:
        """
        self.start_map = start_map
        self.end_map = end_map
        if self.end_map is None:
            self.end_map = self.start_map.duplicate()

        self.transition = transition
        self._is_unique = self.end_map is None or self.transition is None

        if self.transition is None:
            self.transition = FrequencyMapTransition()  # no transition in default

        self._last_freq = None
        self._last_transition_interpolation_points = 0

    @property
    def is_unique(self):
        return self._is_unique

    @is_unique.setter
    def is_unique(self, bool):
        if not bool and (self.end_map is None or self.transition is None):
            raise ValueError('End map and transition must be set for tuple to be non-unique')
        else:
            self._is_unique = bool

    def is_changed(self):
        return self.start_map.is_changed or self.end_map.is_changed or self.transition.is_changed

    def get_freqs(self, nr_points):
        start_freqs = self.start_map.get_map()
        self.start_map.is_changed = False

        if not self.is_unique:
            end_freqs = self.end_map.get_map()
            self.end_map.is_changed = False

            if self._last_transition_interpolation_points != nr_points:
                self.transition.set_interpolation_xs(np.linspace(0, 1, nr_points))
                self._last_transition_interpolation_points = nr_points

            transition_values = self.transition.get_interpolation()[0]
            self.transition.is_changed = False

            freqs = np.einsum('i,j->ij', start_freqs, transition_values) \
                    + np.einsum('i,j->ij', end_freqs, (1 - transition_values))
        else:
            transition_values = np.ones(nr_points)
            freqs = np.einsum('i,j->ij', start_freqs, transition_values)

        return freqs


class FrequencyMapTransition(graph.Graph):
    """
    Transition between two maps.
    """

    def __init__(self, length=10, values=None, interpolation_points=1000):
        if values is None:
            values = np.ones(length)
        graph.Graph.__init__(self, values, np.linspace(0, 1, length), raw_data=False, settable=True,
                             on_get_interpolation_unchanged=False)

        self.set_interpolation_xs(np.linspace(0, 1, interpolation_points))
        self.start_px = 0

    def filter_interpolated(self, ys, xs):
        return np.clip(ys, a_min=0, a_max=1), xs


class FrequencyMapArray(data_utils.StoreInterface):
    """
    A mapping between px in y-direction and frequency that can contain multiple transitions between partial, temporarily
    constrained mappings.
    """

    def __init__(self, track, freq_map=None):
        self.map_tuples = []
        self.track = track

        self.add_tuple(None, freq_map)
        self.last_freqs = None

    def add_tuple(self, tuple, freq_map):
        new_map = freq_map
        if new_map is None:
            new_map = data_structures.scales.LinearFrequencyMap(self.track.height).set().get()
        new_transition = FrequencyMapTransition()

        if tuple is not None:
            new_tuple = FrequencyMapTuple(tuple.start_map, new_map, new_transition)
            ind = self.map_tuples.index(tuple)
            self.map_tuples.insert(ind, new_tuple)
            tuple.start_map = new_map
        else:
            new_start_map = data_structures.scales.LinearFrequencyMap(self.track.height).set().get()
            new_tuple = FrequencyMapTuple(new_start_map, new_map, new_transition)
            self.map_tuples.append(new_tuple)

    def get_freqs(self, nr_samples_per_px):
        if self.is_changed():
            if len(self.map_tuples) == 1:
                nr_time_px = config.params.image_width
                freqs = self.map_tuples[0].get_freqs(nr_time_px * nr_samples_per_px)
                self.last_freqs = data_utils.LoopArray(freqs, np.float64, shared=False, modes=['raise', 'wrap'])

            elif len(self.map_tuples) > 1:
                nr_time_px = self.map_tuples[1].start_px
                freqs = self.map_tuples[0].get_freqs(nr_time_px * nr_samples_per_px)

                for i, tuple in self.map_tuples[1:-1]:
                    nr_time_px = self.map_tuples[i + 1].start_px - self.map_tuples[i].start_px
                    tuple_freqs = tuple.get_freqs(nr_time_px * nr_samples_per_px)
                    freqs = np.concatenate((freqs, tuple_freqs), axis=1)

                tuple = self.map_tuples[-1]
                nr_time_px = self.track.length - tuple.start_px
                tuple_freqs = tuple.get_freqs(nr_time_px * nr_samples_per_px)
                freqs = np.concatenate((freqs, tuple_freqs), axis=1)

                self.last_freqs = data_utils.LoopArray(freqs, np.float64, shared=False, modes=['raise', 'wrap'])

        return self.last_freqs

    def is_changed(self):
        return np.logical_or.reduce(np.array([tuple.is_changed() for tuple in self.map_tuples]))

    def duplicate(self):
        return copy.deepcopy(self)

    def get_preview(self, dims):
        pass


class Timbre(data_utils.StoreInterface):
    """ Timbre holds all details to modulate the phase-to-wavetable translation and transforms them to a form usable
    for the Synthesizer.



    In the moment amplitude modulation specifies
        amplitude and frequency modulation amplitude and frequency (ama, fma, amf, fmf)
        harmonic
        shift
    which are stored in form of a np.array. Multiple timbres can be overlayed for the same track.
    """

    def __init__(self):
        # columns: weight, ama, amf, harmonic, fma, fmf, shift
        self.nr_timbre_params = 7
        self._timbre = np.array([[1, 0, 0, 1, 0, 0, 0]])
        self.names = ['New Partial']

    @property
    def partial_waves(self):
        return self._timbre

    @property
    def shape(self):
        return self._timbre.shape

    def update(self, modulation_options):
        self._timbre = np.zeros((len(modulation_options), self.nr_timbre_params))
        for i, mod_table in enumerate(modulation_options):
            self._timbre[i][0] = mod_table.options['weight']
            self._timbre[i][1] = mod_table.options['ama']
            self._timbre[i][2] = mod_table.options['amf']
            self._timbre[i][3] = mod_table.options['harmonic']
            self._timbre[i][4] = mod_table.options['fma']
            self._timbre[i][5] = mod_table.options['fmf']
            self._timbre[i][6] = mod_table.options['shift']

        return self._timbre

    def duplicate(self):
        return copy.deepcopy(self)

    def get_preview(self, dims):
        pass

