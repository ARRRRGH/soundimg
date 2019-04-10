import run_time.config as config
import run_time.thread as thread
import run_time.audio_export as audio_export
import ui.gtk_utils as gtk_utils
import ui.overviews as overviews
import ui.base_widgets as base_widgets

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GObject

import os
import numpy as np
import pyaudio


class Player(Gtk.Box, gtk_utils.GtkListener, base_widgets.EclipseChild):
    __gsignals__ = {'start_playing': (GObject.SignalFlags.RUN_LAST, None, []),
                    'stop_playing': (GObject.SignalFlags.RUN_LAST, None, [])}

    def __init__(self, thread_manager, params):
        """
        A sound player widget. The widget holds buttons to direct playback and drawing. It is charged of both generating
        and playing sound from a TrackView instance.

        :param builder: Gtk.Builder holding the player interface
        :param thread_manager: all threads generated in the player are added to the thread manager
        """
        Gtk.Box.__init__(self)

        self.params = params
        self.stream = None

        player_file = os.path.join(config.glade_dir, "player.glade")
        builder = Gtk.Builder()
        builder.add_from_file(player_file)

        canvas_field = builder.get_object('canvas_field')
        self.player_toolbar = builder.get_object('player_toolbar')
        self.player_grid = builder.get_object('player_grid')
        self.player_window = builder.get_object('player_window')
        player_box = builder.get_object('player_box')

        self.player_window.remove(player_box)
        self.pack_start(player_box, True, True, 0)

        # for pyaudio_callback
        self.zeros = np.zeros(config.params.block_length * config.params.channels,
                              dtype=config.params.dtype).tostring()
        self.played_samples = 0
        self.time_in_percent = 0 # updated by pyaudio_callback

        self.playing = False

        self.just_muted = True

        self.sound_gen_thread = thread_manager.new_thread(thread.SoundThread,'sound_gen_thread', self.params, self)

        self.view = overviews.TrackView(min_dims=config.dims_main_player_canvas)
        canvas_field.add(self.view)

        builder.connect_signals(self)

    def set_time(self, val):
        self.played_samples = val
        self.time_in_percent = self.get_time_in_percent()

    def get_time_px(self):
        return self.get_absolute_px_from_sample(self.played_samples)

    def get_time_in_percent(self):
        return self.get_absolute_sample_from_sample(self.played_samples) / \
               float(config.params.image_width * self.params.samples_per_px)

    def get_absolute_px_from_px(self, px):
        return px % config.params.image_width

    def get_absolute_px_from_sample(self, sample):
        px = sample // self.params.samples_per_px
        return self.get_absolute_px_from_px(px)

    def get_absolute_sample_from_sample(self, sample):
        return sample % (config.params.image_width * self.params.samples_per_px)

    def set_player_params(self, params):
        self.params = params
        self.sound_gen_thread.set_player_params(params)

    def create_stream(self):
        p = pyaudio.PyAudio()

        stream = p.open(format=pyaudio.paFloat32,
                             channels=config.params.channels,
                             rate=config.params.output_sample_rate,
                             output=True,
                             frames_per_buffer=config.params.block_length,
                             stream_callback=self.pyaudio_callback)

        return stream

    def start_stream(self):
        self.stream = self.create_stream()
        self.stream.start_stream()

    def stop_stream(self):
        if self.stream is not None:
            self.stream.close()

    def pause(self, regenerate, where=None):
        self.playing = False
        if where is None:
            where = self.played_samples
        self.stop_stream()

        self.sound_gen_thread.reset(regenerate=regenerate, where=where)
        self.emit('stop_playing')
        self.view.refresh()

    def play(self):
        self.playing = True
        self.start_stream()
        self.emit('start_playing')

    def set_played_px(self, back=None, forward=None):
        if back is not None and back > 0:
            back *= self.params.samples_per_px
            played_samples = max(self.played_samples - back, 0)
        elif forward is not None and forward > 0:
            forward *= self.params.samples_per_px
            played_samples = self.played_samples + forward
        self.sound_gen_thread.reset(regenerate=False, where=played_samples)

    def pyaudio_callback(self, in_data, sample_count, time_info, flag):
        # if self.sound_gen_thread.is_buffer_filtered():
            # make sure buffer is full when restarting playback
            # if self.just_muted and not self.sound_gen_thread.is_buffer_full():
            #     return self.zeros, pyaudio.paContinue

        self.just_muted = False
        played_samples = self.played_samples
        self.set_time(self.played_samples + sample_count)

        block_length_range = slice(played_samples, self.played_samples)
        return self.sound_gen_thread.buffer.as_buffer(block_length_range), pyaudio.paContinue
        # else:
        #     return self.zeros, pyaudio.paContinue

    def add_track(self, track):
        self.sound_gen_thread.add_track(track)
        self.view.add_track(track)

    def remove_track(self, track):
        self.sound_gen_thread.remove_track(track)
        self.view.remove_track(track)

    def export_to_wav(self):
        file_chooser = Gtk.FileChooserWidget()
        file_chooser.set_action(Gtk.FileChooserAction.SAVE)
        file_chooser.set_do_overwrite_confirmation(True)
        file_chooser.set_current_folder(config.project.dir)

        dialog = base_widgets.ResponseDialog('Export Project to MP3', config.main_window, file_chooser,
                                             ok=False, cancel=True, save=True)

        response = dialog.run()
        if response == Gtk.ResponseType.ACCEPT:
            path = os.path.join(file_chooser.get_current_folder(), file_chooser.get_current_name())
            if not path.endswith('.wav'):
                path = os.path.join('.wav')

            audio_export.export_to_wav(path,
                                       self.sound_gen_thread.buffer.arr,
                                       self.params.output_sample_rate,
                                       self.params.channels)

        dialog.destroy()


    ######### UI signals ###############################################################################################
    def on_play_button_clicked(self, sender, data=None):
        if self.playing:
            self.pause(regenerate=False)
        else:
            self.play()

    def on_forward_button_clicked(self, sender, data=None):
        px_forward = 10
        self.set_played_px(forward=px_forward)
        self.view.refresh()

    def on_backward_button_clicked(self, sender, data=None):
        px_back = 10
        self.set_played_px(back=px_back)
        self.view.refresh()

    def on_reset_button_clicked(self, sender, data=None):
        config.instrument_overview.update()
        self.pause(regenerate=True, where=0)

    def on_restart_button_clicked(self, sender, data=None):
        self.pause(regenerate=False, where=0)

    def on_select_button_clicked(self, sender, data=None):
        pass

    def on_reset_track(self, source, track):
        self.sound_gen_thread.track_generators[track].reset_to_regenerate()


class SpinnerOnSignal(Gtk.Spinner):
    def __init__(self, source, on_signal, off_signal):
        Gtk.Spinner.__init__(self)
        source.connect(on_signal, self.on_on_signal)
        source.connect(off_signal, self.on_off_signal)

    def on_on_signal(self, source):
        self.start()

    def on_off_signal(self, source):
        self.stop()


class OptionsWindow(Gtk.VBox, base_widgets.EclipseChild):
    def __init__(self, dims=None):
        Gtk.VBox.__init__(self)

        if dims is not None:
            self.set_size_request(dims[1], dims[0])

        self.stack = Gtk.Stack()
        self.stack.set_vhomogeneous(False)

        self.stack_switcher = Gtk.StackSwitcher()
        self.stack_switcher.set_stack(self.stack)

        self.pack_start(self.stack, True, True, 0)
        self.valign = Gtk.Align.START

        self.option_toolbar = Gtk.HBox()
        self.option_toolbar.pack_start(self.stack_switcher, False, False, 0)

        # Instrument Page
        self.instrument_overview = overviews.InstrumentOverview(tracks=[], min_dims=config.dims_main_player_canvas)
        config.instrument_overview = self.instrument_overview

        # Post Synthesis Page
        self.buffer_overview = overviews.BufferOverview(tracks=[],
                                              buffers=[],
                                              sound_gen_thread=config.player.sound_gen_thread,
                                              min_dims=config.dims_main_player_canvas)
        config.buffer_overview = self.buffer_overview

        # Frequency Map Page
        self.freq_map_overview = overviews.FrequencyMapOverview([], min_dims=config.dims_main_player_canvas)
        config.freq_map_overview = self.freq_map_overview

        # add pages
        self.add_page(self.instrument_overview, 'Instrument')
        self.add_page(self.freq_map_overview, 'Frequency Map')
        self.add_page(self.buffer_overview, 'Post Synthesis')

    def add_page(self, page_overview, name):
        self.stack.add_titled(page_overview, name, name)

    def remove_track(self, track):
        ovs = self.instrument_overview, self.buffer_overview, self.freq_map_overview
        for ov in ovs:
            ov.remove_track(track)

    def add_track(self, track):
        # update overviews
        self.instrument_overview.add_track(track, min_dims=config.dims_main_player_canvas)
        self.buffer_overview.add_track(track,
                                       config.player.sound_gen_thread.track_generators[track].buffer,
                                       config.player.sound_gen_thread,
                                       min_dims=config.dims_main_player_canvas)
        self.freq_map_overview.add_track(track, min_dims=config.dims_main_player_canvas)

    def hide(self):
        super(OptionsWindow, self).hide()
        self.stack_switcher.hide()

    def show(self):
        super(OptionsWindow, self).show()
        self.stack_switcher.show()

    def eclipse_remove(self, parent):
        parent.remove(self)
        self.stack_switcher.hide()

    def eclipse_show(self, parent):
        parent.pack_start(self, True, True, 0)
        self.stack.show()
        # self.stack_switcher.show_all()
        self.show()


class Table(base_widgets.EclipseWindow):
    def __init__(self, project, dims=None):
        self.project = project

        # initialize main_player
        self.player = Player(config.thread_manager, self.project.params.sound_gen_params)
        config.signal_manager.add_signal('start_playing', self.player)
        config.signal_manager.add_signal('stop_playing', self.player)

        config.player = self.player

        # initialize options field
        self.options = OptionsWindow(dims=config.dims_main_player_canvas)

        add_track_button = Gtk.Button('+')
        add_track_button.connect('clicked', self.on_add_track_clicked)
        self.option_toolbar.pack_end(add_track_button, False, False, 2)

        base_widgets.EclipseWindow.__init__(self, self.player, self.options, self.options.option_toolbar, dims=dims)

        self.initialize_tracks_from_project()

    @property
    def option_toolbar(self):
        return self.options.option_toolbar

    def initialize_tracks_from_project(self):
        for track in self.project.data.tracks:
            self.add_track(track)

    def on_add_track_clicked(self, source):
        # create new track
        track = self.project.data.add_track(config.project.params)
        if track is not None:
            self.add_track(track)

    def on_remove_track(self, source, track):
        config.project.data.tracks.remove(track)
        self.player.remove_track(track)
        self.options.remove_track(track)
        config.signal_manager.clear_from_key(track)

    def add_track(self, track):
        self.player.add_track(track)
        self.options.add_track(track)

        config.signal_manager.connect('reset_track', self.player.on_reset_track)
        config.signal_manager.connect('remove_track', self.on_remove_track)
        config.brush_window.emit('set-new-current-brush', config.brush_window.current_brush)
