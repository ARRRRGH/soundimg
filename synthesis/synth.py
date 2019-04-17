import data_structures.data_utils as data_utils
import data_structures.image as image
import run_time.config as config
import run_time.py_utils as py_utils

# import cv2

from numba import jit, prange
import numpy as np


class TrackGenerator(object):
    def __init__(self, track):
        self.track = track

        self.params = None
        self.playable = False
        self.buffer = None

        self.playable_image = image.PlayableImage(track)
        self.synthesizer = None

    def set_play_params(self, params):
        self.params = params
        self.playable = True

        self.playable_image.reset()
        self.buffer = data_utils.Buffer(channels=config.params.channels,
                                        length=config.params.image_width * self.params.samples_per_px,
                                        dtype=config.params.dtype,
                                        shared=False)

        self.synthesizer = Synthesizer(self.track.instrument, params, self.track.freq_map_array, self.track.limiter)

    def set_init_conditions(self, *args, **kwargs):
        self.playable_image.set(*args, **kwargs)

    def reset_init_conditions(self, *args, **kwargs):
        self.playable_image.reset()

    def reset_state_col(self, *args, **kwargs):
        self.playable_image.image.set_state_col(*args, **kwargs)

    def is_changed_in_range(self, *args, **kwargs):
        return self.playable_image.image.is_changed_in_range(*args, **kwargs)

    def get_next_changed_px(self, *args, **kwargs):
        return self.playable_image.image.get_next_changed_px(*args, **kwargs)

    def gen_sound(self, px_range):
        data, sample_range = self.playable_image.gen_sound(self.synthesizer, px_range)
        self.buffer[sample_range] = data
        return sample_range

    def reset_to_regenerate(self):
        if self.playable:
            self.reset_init_conditions()
            self.reset_state_col(1)
            self.buffer.reset()


class Synthesizer(object):
    def __init__(self, instrument, player_params, freq_map_array, limiter=None):
        self.instrument = instrument
        self.limiter = limiter
        self.freq_map_array = freq_map_array
        self.params = player_params

    def set_play_params(self, play_params):
        self.params = play_params

    def generate_sound_from_image(self, imag, px_range, init_conditions):
        """
        Generates sound from image within the limits given by px_range and assuming initial conditions
        init_conditions.

        :param      image: the image to convert to sound
        :param      px_range: the specific pixel range (slice for y, slice for x) that is converted
        :param      init_conditions: tuple (outer phase, inner phase, final cond of last freq spectr)

        :return:    sound:  arr of length len(px_range[1]) * samples_per_px
                    sample_range_out: explicit range for buffer
                    final conditions: tuple (final outer phase, final inner phase, final column freq_spectr)
        """
        sample_range_out = slice(px_range.start * self.params.samples_per_px,
                                 px_range.stop * self.params.samples_per_px)

        # include one px more to anticipate drop
        px_range = slice(px_range.start, px_range.stop + 1)
        sample_range_in = slice(px_range.start * self.params.samples_per_px,
                                px_range.stop * self.params.samples_per_px)

        lr_freq_spectrs = self.freq_spectr_per_channel(imag, px_range)

        final_conditions = []
        channel_sounds = []
        for i in range(config.params.channels):
            channel_init_conditions = init_conditions[i]
            sound, f_conds = self.synthesize_image(lr_freq_spectrs[i], sample_range_in, channel_init_conditions)

            final_conditions.append(f_conds)
            channel_sounds.append(sound)

        return channel_sounds, sample_range_out, final_conditions

    def freq_spectr_per_channel(self, imag, px_range):
        freq_spectr_pxs = []

        for channel in range(config.params.channels):
            # read image
            freq_spectr_px = self._read_from_image(imag, px_range, channel)

            # apply limiter
            freq_spectr_pxs.append(freq_spectr_px)

        if self.limiter is not None:
            envelope = self.limiter.envelope
            freq_spectr_pxs = _apply_limiter(freq_spectr_pxs, envelope[px_range])

        return freq_spectr_pxs

    def synthesize_image(self, freq_spectr_px, sample_range_in, init_conditions):
        # find rise, drop and transition
        mask = _find_patterns(freq_spectr_px, init_conditions[2])

        # enlarge to sampling rate
        # freq_spectr_samples = cv2.resize(freq_spectr_px,
        #                                  (freq_spectr_px.shape[1] * self.params.samples_per_px,
        #                                   freq_spectr_px.shape[0]),
        #                                  interpolation=cv2.INTER_AREA)

        freq_spectr_samples = py_utils.ImageHandler.resize(freq_spectr_px,
                                                           (freq_spectr_px.shape[1] * self.params.samples_per_px,
                                               freq_spectr_px.shape[0]))

        freqs_samples = self.freq_map_array.get_freqs(self.params.samples_per_px)[:, sample_range_in]

        # # each image has its own freqs
        # freqs_px = freqs[:, px_range]
        #
        # # enlarge to sampling rate
        # freqs_samples = cv2.resize(freqs_px,
        #                            (freqs_px.shape[1] * self.params.samples_per_px, freqs_px.shape[0]),
        #                            interpolation=cv2.INTER_AREA)

        # get sound and final phases
        sound, final_outer_phase, final_inner_phases = self._generate_sound_wave(freqs_samples, freq_spectr_samples,
                                                                                 mask, init_conditions)

        # exclude again last px column in freq_spectr that was included only to anticipate drop
        last_freq_spectr = freq_spectr_samples[:, -self.params.samples_per_px - 1]

        # packing final conditions
        final_conditions = (final_outer_phase, final_inner_phases, last_freq_spectr)

        return sound, final_conditions

    def _read_from_image(self, imag, px_range, channel):
        """
        Reads out the values from image within the limits given by px_range.

        :param imag:
        :param px_range:
        :return: np.array, frequency spectrum globally normalized
        """
        img = imag[:, px_range, :]

        # differentiate red / blue if there are two channels
        if config.params.channels == 2:
            if channel == 1:
                freq_spectr = np.sum(img[:, :, 0:2], axis=2) / 510.
            else:
                freq_spectr = np.sum(img[:, :, 1:3], axis=2) / 510.

        # if it's mono, take the whole brightness
        else:
            freq_spectr = np.sum(img, axis=2) / 765.

        return freq_spectr

    # @jit(cache=False, parallel=True, fastmath=True)
    def _generate_sound_wave(self, freqs, freq_spectr_samples, mask, init_conditions):
        """
        Auxiliary function including the preparation of the frequency spectrum (rise, drop, transition) and its
        reduction to a sound wave.

        :param freqs:
        :param freq_spectr_samples:
        :param mask:
        :param init_conditions:
        :return:
        """
        # pack parameters needed in numba functions
        fundamental_freq = get_fundamental_freq(self.instrument.wave_table, config.params.output_sample_rate)

        params = (self.params.samples_per_px,
                  self.params.output_sample_rate,
                  fundamental_freq,
                  self.instrument.wave_table.length)

        # create smooth rise and drops in freq_spectr_samples from mask
        outer_phases, inner_phases = _prepare_phases(freq_spectr_samples,
                                                     mask,
                                                     init_conditions,
                                                     freqs,
                                                     self.instrument.osc_params,
                                                     params)

        freq_spectr_samples = _prepare_freq_spectr(freq_spectr_samples,
                                                   mask,
                                                   init_conditions,
                                                   self.instrument.fade_arrs,
                                                   params)

        # exclude again last column in freq_spectr that was included only to anticipate drop
        freq_spectr_samples = freq_spectr_samples[:, :-self.params.samples_per_px]

        # read out samples from wave_table (for each entry in timbre)
        wav = self.instrument.wave_table.get_ind(inner_phases)

        # multiply samples times the frequency spectrum
        for i in range(wav.shape[2]):
            wav[:, :, i] *= freq_spectr_samples

        # sum along frequency axis
        wav = np.sum(wav, axis=0)

        # amplitude modulation per timbre and summing along _timbre axis
        wav = _amp_modulation(wav, self.instrument.osc_params, outer_phases)
        return wav, outer_phases[-1], inner_phases[:, -1]


################### CALC ######################################################

# @jit(nopython=True, fastmath=True, parallel=True)
def _apply_limiter(freq_spectrs, limiter):
    tot_freq_spectr = np.sum(freq_spectr for freq_spectr in freq_spectrs)
    norm = np.sum(tot_freq_spectr, axis=0)
    inds_exceeding = np.where(norm > limiter)[0]
    for freq_spectr in freq_spectrs:
        freq_spectr[:, inds_exceeding] *= limiter[inds_exceeding] / norm[inds_exceeding]
    return freq_spectrs


@jit(cache=False, fastmath=True, nopython=True)
def _amp_modulation(wav, osc_params, outer_phases):
    """
    Modulates the amplitude of an ensemble of n partial sound waves with m samples packed in an array with shape (m, n)
    accornding to the oscillation parameters in osc_params. Returns the sum of the modulated partial waves.
    The amplitude modulation oscillates with outer_phases.

    :param wav:
    :param osc_params:
    :param outer_phases:
    :return:
    """
    ret = np.zeros(len(outer_phases))
    for i in prange(osc_params.shape[0]):
        # fixme: fixed to outer_phases[0, :]
        ret += _amp_modulation_per_timbre(wav[:, i], osc_params[i], outer_phases)
    return ret


@jit(cache=False, nopython=True, fastmath=True, parallel=True)
def _amp_modulation_per_timbre(wav, osc_params, outer_phases):
    return wav * (osc_params[1] * (np.sin(osc_params[7] + outer_phases * osc_params[3]) + 1.) + osc_params[2])


@jit(cache=False, nopython=True, fastmath=True, parallel=True)
def _freq_modulation(freqs, osc_params, outer_phases):
    """
    Modulates the frequency of an ensemble of n partial sound waves according to osc_params. Returns n np.arrays
    containing the modulated frequencies.

    :param freqs:
    :param osc_params:
    :param outer_phases:
    :return:
    """
    ret = np.zeros((freqs.shape[0], osc_params.shape[0]))
    for j in prange(ret.shape[1]):
        for i in prange(ret.shape[0]):
            ret[i, j] = freqs[i] * osc_params[j, 4] \
                        * (1 + osc_params[j, 5] * np.sin(osc_params[j, 7] + outer_phases[i] * osc_params[j, 6]))

    return ret


@jit(cache=False, fastmath=True, parallel=True, nopython=True, nogil=True)
def _prepare_phases(freq_spectr, mask, init_conditions, freqs, osc_params, params):
    samples_per_px, output_sample_rate, fundamental_freq, wav_length = params
    init_outer_phases, init_inner_phases, last_freq_spectr = init_conditions

    # outer phases
    # prepare phases, first column holds init_phase, so enlarge by one
    length = freq_spectr.shape[1] + 1
    outer_phases = init_outer_phases + np.arange(1, length + 1) * 2 * np.pi / output_sample_rate

    # inner phases
    # prepare phases, first column holds init_phase, so enlarge by one
    inner_phases = np.zeros((freq_spectr.shape[0], freq_spectr.shape[1] + 1, osc_params.shape[0]))
    inner_phases[:, 0, :] = init_inner_phases

    for j in range(mask.shape[1] - 1):
        # map px onto samples by j -> j_s
        j_s = j * samples_per_px
        next_j_s = j_s + samples_per_px
        for i in range(mask.shape[0]):
            # load freqs, needed for inner_phases
            fs = freqs[i, j_s:next_j_s]

            # create phases when there is sound
            if not (mask[i, j] == 0 and freq_spectr[i, j_s] == 0):
                # j_s + 1 is current index as inner_phases and outer_phases are shifted by one column due to the
                # inclusion of the initial condition
                inner_phases[i, j_s + 1:next_j_s + 1, :] = inner_phases[i, j_s, :] \
                                                           + _get_inner_phases(fs,
                                                                               outer_phases[j_s + 1:next_j_s + 1],
                                                                               osc_params,
                                                                               fundamental_freq)

    # remove initial phases from array
    # remove last px due to freq_spectr.px_range[1] being larger by one
    outer_phases = outer_phases[1: -samples_per_px]
    inner_phases = inner_phases[:, 1: -samples_per_px, :]

    return outer_phases, inner_phases


# @jit(cache=False, fastmath=True, parallel=True, nopython=True)
def _prepare_freq_spectr(freq_spectr, mask, init_conditions, fade_arrs, params):
    """
    Prepares the frequency spectrum by setting sub-pixel information in freq_spectr.

    :param freq_spectr:
    :param init_conditions:
    :param mask:
    :param freqs:
    :param osc_params:
    :param params:
    :return:
    """
    # unpacking params
    samples_per_px, output_sample_rate, fundamental_freq, wav_length = params
    init_outer_phases, init_inner_phases, last_freq_spectr = init_conditions
    fade_in_arr, fade_out_arr = fade_arrs

    transition_length_out = len(fade_out_arr)
    transition_length_in = len(fade_in_arr)

    # iterate over px, excluding last column which is just anticipation and will not be used for sound generation
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1] - 1):

            # map px onto samples by j -> j_s
            j_s = j * samples_per_px
            # next_j_s = j_s + samples_per_px

            # rise
            if mask[i, j] == 1:
                transition_end_j_s = j_s + transition_length_in

                if j_s == 0:
                    vol_ini = last_freq_spectr[i]
                else:
                    vol_ini = freq_spectr[i, j_s - 1]
                vol_end = freq_spectr[i, transition_end_j_s]

                # freq_spectr[i] = filters.fade_in(j_s, transition_end_j_s, vol_ini, vol_end, freq_spectr[i])
                freq_spectr[i, j_s:transition_end_j_s] = vol_ini + fade_in_arr * (vol_end - vol_ini)

            # drop
            elif mask[i, j] == -1 and j != 0:  # j == 0 drops are rendered in delay before, see case j = len(mask[i])-1
                transition_start_j_s = j_s - transition_length_out  # is necessarily > 0 since j >= 1

                if transition_start_j_s == 0:
                    vol_ini = last_freq_spectr[i]
                else:
                    vol_ini = freq_spectr[i, transition_start_j_s - 1]
                vol_end = freq_spectr[i, j_s]

                # freq_spectr[i] = filters.fade_out(transition_start_j_s, j_s, vol_ini, vol_end, freq_spectr[i])
                freq_spectr[i, transition_start_j_s:j_s] = vol_end - fade_out_arr * (vol_end - vol_ini)

        # consider last column, only drop relevant as rise is not rendered in retrospect
        j = len(mask[i]) - 1
        if mask[i, j] == -1:
            j_s = j * samples_per_px
            transition_start_j_s = j_s - transition_length_out
            if transition_start_j_s == 0:
                vol_ini = last_freq_spectr[i]
            else:
                vol_ini = freq_spectr[i, transition_start_j_s - 1]
            vol_end = freq_spectr[i, j_s]

            # freq_spectr[i] = filters.fade_out(transition_start_j_s, j_s, vol_ini, vol_end, freq_spectr[i])
            freq_spectr[i, transition_start_j_s:j_s] = vol_end - fade_out_arr * (vol_end - vol_ini)

    return freq_spectr


@jit(cache=False, fastmath=True, nopython=True)
def _find_patterns(freqSpectr, last_freq_spectr):
    """
    Creates a mask containing information of rises, drops and transitions in freqSpectr. The mask is an array with
    entries describing if the value in this px of freqSpectr is
        1: higher than in the one before,
       -1: smaller than in the one before,
        0: the same as before

    :param freqSpectr:
    :param last_freq_spectr:
    :return:
    """
    # include last column of last_freq_spectr as 'initial condition'
    shape = (freqSpectr.shape[0], freqSpectr.shape[1] + 1)
    freq_spectr_with_last = np.zeros(shape)
    freq_spectr_with_last[:, 1:] = freqSpectr
    freq_spectr_with_last[:, 0] = last_freq_spectr

    # fixme: array creation really needed?
    shifted_left = np.zeros(shape)
    shifted_left[:, :-1] = freq_spectr_with_last[:, 1:]

    mask_rise = (freq_spectr_with_last < shifted_left) * 1
    mask_drop = (freq_spectr_with_last > shifted_left) * 1

    mask = mask_rise - mask_drop  # never rise and drop in one px
    return mask[:, :-1]  # because of initially included initial condition


@jit(cache=False, fastmath=True, nopython=True)
def _get_inner_phases(fs, outer_phases, osc_params, fundamental_freq):
    f = _freq_modulation(fs, osc_params, outer_phases)
    steps = np.floor(f / fundamental_freq)
    # reuse array
    ind = f
    ind[0, :] = steps[0, :]
    for ii in range(1, ind.shape[0]):
        ind[ii, :] = ind[ii - 1, :] + steps[ii, :]

    return ind


def get_fundamental_freq(wt, output_sample_rate):
    return float(output_sample_rate) / wt.length
