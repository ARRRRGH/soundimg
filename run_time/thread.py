#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Refactored by Malcolm Jones to work with GTK+3 PyGobject( aka PyGI ). Mar 2016.

# Demo application showing how once can combine the python
# threading module with GObject signals to make a simple thread
# manager class which can be used to stop horrible blocking GUIs.
#
# (c) 2008, John Stowers <john.stowers@gmail.com>
#
# This program serves as an example, and can be freely used, copied, derived
# and redistributed by anyone. No warranty is implied or given.

import run_time.py_utils as py_utils
import run_time.config as config
import data_structures.data_utils as data_utils
import synthesis.filters as filters
import synthesis.synth as synth
import ui.gtk_utils as gtk_utils

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import GObject

import traceback
import numpy as np
import threading
import time


class PeriodicFunctionCall(object):
    def __init__(self, interval, target, deamon):
        self.interval = interval
        self.target = target
        self.deamon = deamon
        self.stopped = False
        self._thread = None

    def start(self):
        self.stopped = False
        self._thread = threading.Thread(target=self.do_target)
        self._thread.deamon = self.deamon
        self._thread.start()

    def stop(self):
        self.stopped = True

    def do_target(self):
        while not self.stopped:
            self.target()
            time.sleep(self.interval)


class _Thread(threading.Thread, gtk_utils._IdleObject):
    """
    Cancellable thread which uses gobject signals to return information
    to the GUI.
    """
    __gsignals__ = {
        "completed": (
            GObject.SignalFlags.RUN_LAST, None, []),
        "progress": (
            GObject.SignalFlags.RUN_LAST, None, [
                GObject.TYPE_FLOAT])  # percent complete
    }

    # @trace
    def __init__(self, thread_name, *args):
        threading.Thread.__init__(self)
        gtk_utils._IdleObject.__init__(self)
        self.cancelled = False
        self.thread_name = thread_name
        self.name = thread_name

    # @trace
    def cancel(self):
        """
        Threads in python are not cancellable, so we implement our own
        cancellation logic
        """
        self.cancelled = True


class ThreadManager:
    """
    Manages many FooThreads. This involves starting and stopping
    said threads, and respecting a maximum num of concurrent threads limit
    """

    # @trace
    def __init__(self, maxConcurrentThreads):
        self.maxConcurrentThreads = maxConcurrentThreads
        # stores all threads, running or stopped
        self.gtkThreads = {}
        # the pending thread args are used as an index for the stopped threads
        self.pendingGtkThreadArgs = []

    # @trace
    def _register_thread_completed(self, thread, *args):
        """
        Decrements the count of concurrent threads and starts any
        pending threads if there is space
        """
        del (self.gtkThreads[thread.thread_name])
        running = len(self.gtkThreads) - len(self.pendingGtkThreadArgs)

        if running < self.maxConcurrentThreads:
            try:
                name = self.pendingGtkThreadArgs.pop()
                self.gtkThreads[name].start()
            except IndexError:
                pass

    # @trace
    def new_thread(self, classThread, thread_name, *args):
        """
        Makes a thread with args. The thread will be started when there is
        a free slot
        """
        running = len(self.gtkThreads) - len(self.pendingGtkThreadArgs)

        if args not in self.gtkThreads:
            thread = classThread(thread_name, *args)
            # signals run in the order connected. Connect the user completed
            # callback first incase they wish to do something
            # before we delete the thread
            thread.connect("completed", self._register_thread_completed, *args)
            # This is why we use args, not kwargs, because args are hashable
            self.gtkThreads[thread_name] = thread

            if running < self.maxConcurrentThreads:
                self.gtkThreads[thread_name].start()
            else:
                self.pendingGtkThreadArgs.append(thread.thread_name)

            return thread

        return None

    def get_thread_by_name(self, thread_name):
        return self.gtkThreads[thread_name]

    # @trace
    def stop_all_threads(self, block=False):
        """
        Stops all threads. If block is True then actually wait for the thread
        to finish (may block the UI)
        """
        for thread in self.gtkThreads.values():
            thread.cancel()
            if block:
                if thread.isAlive():
                    thread.join()

    def is_active(self):
        for thread in self.gtkThreads.values():
            if thread.isAlive():
                return True
        return False


class SoundThread(_Thread):

    __gsignals__ = py_utils.merge_two_dicts({'not_idle': (GObject.SignalFlags.RUN_LAST, None, []),
                                          'idle': (GObject.SignalFlags.RUN_LAST, None, []),
                                          }, _Thread.__gsignals__)

    def __init__(self, thread_name, params, player):
        """
        A thread that directs sound generation based on params.

        :param thread_name: str
        :param args: player instance
        """
        _Thread.__init__(self, thread_name)

        self.track_generators = {}

        self.player = player
        self.params = params
        self.set_play_params(self.params)

        self._generated_px = 0  
        self._filtered_px = 0
        self.is_idle = True

        # params set once per audio gen loop
        self.time_buffer = 2
        self.num_px_buffer = int(self.time_buffer / self.params.time_per_px)
        self.num_samples_buffer = self.num_px_buffer * self.params.samples_per_px

        self.num_delays_per_buff = self.num_px_buffer // 20
        self.num_px_delay = max(int(self.num_px_buffer // self.num_delays_per_buff), 1)
        self.num_samples_delay = self.num_px_delay * self.params.samples_per_px

        self.buffer = data_utils.Buffer(channels=config.params.channels,
                                        length=config.params.image_width * self.params.samples_per_px,
                                        dtype=config.params.dtype,
                                        shared=False)

        self.delayed_actions = []

    def reset(self, regenerate=False, where=0):
        len_in_samples = config.params.image_width * self.params.samples_per_px

        self._generated_px = where % len_in_samples
        self._filtered_px = where % len_in_samples

        self.player.playing = False
        self.player.just_muted = True
        self.player.set_time(where % len_in_samples)

        if regenerate:
            self.buffer.reset()
            for generator in self.track_generators.values():
                generator.reset_to_regenerate()

    def do_delayed_actions(self):
        for action in self.delayed_actions:
            func, args, kwargs = action
            func(*args, **kwargs)
            self.delayed_actions.remove(action)

    def run(self):
        self.update_buffer()

        self._quit()

    def set_up(self):
        self.set_muted()

    def switch_idle_state(self, is_idle):
        if self.is_idle and not is_idle:
            self.is_idle = is_idle
            self.emit('not_idle')
        elif not self.is_idle and is_idle:
            self.is_idle = is_idle
            self.emit('idle')

    def update_buffer(self):
        try:
            self.set_up()
            while not self.cancelled:
                if self.is_all_generated() and self.is_next_change_ahead():
                    self.switch_idle_state(True)
                    self.do_delayed_actions()
                    time_per_delay = self.params.time_per_px * self.num_px_delay
                    time.sleep(time_per_delay)
                elif self.player.playing:
                    self.generate_new_samples(self.num_px_delay)
                    self.filter_samples()
                    self.player.just_muted = False
                elif not self.player.just_muted:
                    self.set_muted(self._generated_px)
                    # self.just_muted = True
                else:
                    self.generate_new_samples(self.num_px_delay)
                    self.filter_samples()

            return True

        except Exception as e:
            traceback.print_exc()
            print()
            raise e

    def is_buffer_filtered(self):
        return self._filtered_px * self.params.samples_per_px \
               - self.player.played_samples > config.params.block_length

    def is_buffer_full(self):
        return self._generated_px * self.params.samples_per_px \
               - self.player.played_samples > self.num_samples_buffer

    def is_all_generated(self):
        return self._generated_px * self.params.samples_per_px \
               - self.player.played_samples >= 2 * config.params.image_width * self.params.samples_per_px

    def is_next_change_ahead(self):
        return self.get_next_changed_px(none=self._generated_px + 1) > self._generated_px

    def set_muted(self, start_px=None):
        self.set_init_conditions()
        self.player.just_muted = True

    def update_indices(self, delay=1):
        self._generated_px += delay * self.num_px_delay

    def generate_new_samples(self, generate_nr_px):
        x = self.get_next_changed_px()
        if x is not None:
            px_range = slice(x, x + generate_nr_px)
            is_changed, changed_tracks = self.is_changed_in_range(px_range, get_changed_tracks=True)
            self.switch_idle_state(False)
            self.gen_sound_and_merge(px_range, changed_tracks)
            self.update_indices()
        else:
            self.update_indices()

    def _quit(self):
        # pickle.dump(self.buffer.arr, open("save.p", "wb"))
        pass

    # fixme: shift to view class
    def filter_samples(self):
        if self._filtered_px < self._generated_px:
            # samples_range = slice(self._filtered_px * config.params.samples_per_px,
            #                       self._generated_px * config.params.samples_per_px)
            #
            # self.buffer[samples_range] = self._apply_filter(self.buffer, samples_range)
            self._filtered_px = self._generated_px

    # fixme: this is a temporary fix until filtering of samples is implemented
    def _apply_filter(self, sound, samples_range):
        return sound[samples_range]

    def gen_sound_and_merge(self, px_range, changed_tracks):
        data, sample_range = self.gen_sound_from_changed_tracks(px_range, changed_tracks)
        self.buffer[sample_range] = data

        # smooth transition over discontinuity due to phase shift with respect to the stored initial conditions
        is_changed = self.is_changed_in_range(int(px_range.stop))
        if not is_changed:
            transition_length_px = 1
            transition_length_samples = self.params.samples_per_px * transition_length_px

            next_px_slice = slice(px_range.stop, px_range.stop + transition_length_px)

            next_px_data, next_sample_range = self.gen_sound_from_changed_tracks(next_px_slice, changed_tracks)

            transition_slice = slice(sample_range.stop, sample_range.stop + transition_length_samples)
            transition_slice_next = slice(0, config.params.channels * transition_length_samples)

            old = self.buffer[transition_slice]
            new = next_px_data[transition_slice_next]
            self.buffer[transition_slice] = filters.blend(new, old)

    def get_next_changed_px(self, none=None):
        played_px = (self.player.played_samples // self.params.samples_per_px) % config.params.image_width
        generated_px = self._generated_px % config.params.image_width
        current_index = max(played_px + self.num_px_buffer, generated_px)
        # current_index = played_px

        pxs = []
        # fixme: this should be multiprocessed
        for generator in self.track_generators.values():
            px = generator.get_next_changed_px(current_index)
            if px is not None:
                pxs.append(px)

        if len(pxs) == 0:
            next = None
        else:
            next = min(pxs)

        if none is not None and next is None:
            return none
        return next

    def set_play_params(self, play_params):
        self.params = play_params
        self.playable = True

        for generator in self.track_generators.values():
            generator.set_play_params(play_params)

    def set_init_conditions(self):
        for generator in self.track_generators.values():
            generator.reset_to_regenerate()

    def is_changed_in_range(self, key, get_changed_tracks=False):
        ret = 0
        if not get_changed_tracks:
            for generator in self.track_generators.values():
                ret += generator.is_changed_in_range(key)
            return ret
        else:
            changed_tracks = []
            for generator in self.track_generators.values():
                is_changed = generator.is_changed_in_range(key)
                if is_changed:
                    changed_tracks.append(generator)
                ret += is_changed

            return ret, changed_tracks

    def gen_sound_from_changed_tracks(self, px_range, changed_tracks):
        # fixme: this should be multiprocessed
        min_sample = 1e18
        max_sample = 0

        for generator in changed_tracks:
            sample_range = generator.gen_sound(px_range)

            if sample_range.start < min_sample:
                min_sample = sample_range.start
            if sample_range.stop > max_sample:
                max_sample = sample_range.stop

        ret_data = np.sum(generator.buffer[min_sample:max_sample] for generator in self.track_generators.values())
        ret_sample_range = slice(min_sample, max_sample)

        return ret_data, ret_sample_range

    def is_changed(self):
        return self.is_changed_in_range(slice(0, self.shape[1]))

    def add_track(self, track):
        generator = synth.TrackGenerator(track)
        self.track_generators[track] = generator
        generator.set_play_params(self.params)

    def remove_track(self, track):
        self.delayed_actions.append((self._remove_track, [track], {}))

    def _remove_track(self, track, *args, **kwargs):
        del self.track_generators[track]

    def set_player_params(self, params):
        for generator in self.track_generators.values():
            self.params = params
            generator.set_play_params(params)