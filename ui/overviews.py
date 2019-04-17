import run_time.config as config
import ui.imaging as imaging
import ui.base_widgets as base_widgets
import ui.editors as editors
import ui.gtk_utils as gtk_utils

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GObject

import os


class TrackView(base_widgets.CollapsingSectionWindow):
    def __init__(self, *args, **kwargs):
        """
        Image with functionality to convert itself to sound. Sound generation is only possible after set_play_params
        was called.

        TrackView can hold multiple tracks (images) in arr_of_images each associated to an instrument in instruments.

        :param arr_of_tracks:
        :param instruments:
        :param play_params:
        :param args: for image instantiation
        :param kwargs: for image instantiation
        """
        base_widgets.CollapsingSectionWindow.__init__(self, *args, **kwargs)

        # cannot be played unless a params is supplied
        self.params = None
        self.playable = False

        self.current_brush = None

        self.track_panes = {}

    def refresh(self):
        for pane in self.track_panes.values():
            pane.refresh()

    def remove_track(self, track):
        self.remove_child(self.track_panes[track])
        del self.track_panes[track]

    def add_track(self, track, *args, **kwargs):
        track_pane = _TrackPane(track, *args, **kwargs)
        self.add_child(track_pane, config.track_pane_border, with_header=False)

        self.track_panes[track] = track_pane

        # connect current brush of track to this current brush in order to have the same brush for all tracks
        config.signal_manager.connect('set-new-current-brush', self.on_new_current_brush)

        self.collapse_tracks()
        track_pane.magnify()

        self.scrolled_window.scroll_to_child(track)
        self.show_all()

    def collapse_tracks(self):
        self.collapse_children()

    def shrink_tracks(self):
        for track_pane in self.children:
            track_pane.shrink()

    def reset(self, *args, **kwargs):
        for track_pane in self.children:
            track_pane.reset(*args, **kwargs)

    def resize_tracks(self, dims):
        for track_pane in self.children:
            track_pane.resize(dims_canvas=dims)

    def on_new_current_brush(self, source, new_brush):
        self.set_track_brushes(new_brush)

    def set_track_brushes(self, new_brush):
        for track_pane in self.children:
            track_pane.canvas.current_brush = new_brush


class _TrackPane(Gtk.Fixed):
    def __init__(self, track, *args, **kwargs):
        Gtk.Fixed.__init__(self)

        self.name = 'TrackPane'
        if 'name' in kwargs:
            self.name = kwargs['name']

        self.info_bar = _TrackInfoBar(track, config.track_pane_width)

        self.info_bar.connect('open_or_close_track', self.on_open_or_close)
        self.info_bar.connect('resize_track', self.on_resize)

        config.signal_manager.add_signal('reset_track', self.info_bar, key=track)
        config.signal_manager.add_signal('remove_track', self.info_bar, key=track)

        dims_canvas = (config.track_pane_height_opened, config.track_pane_width)
        self.canvas = imaging.Canvas(imag=track.image, dims_canvas=dims_canvas, *args, **kwargs)

        self.track_pane = Gtk.VBox()
        self.track_pane.pack_start(self.info_bar, False, False, 0)
        self.track_pane.add(self.canvas)
        self.add(self.track_pane)

        self.time_pointer = imaging.TimePointer(self.canvas, config.player, config.params.image_width)

        self.is_open = True
        self.is_shrunk = False

        self.params = None
        self.playable = False

        self.track = track

    def refresh(self):
        self.canvas.refresh()
        self.time_pointer.refresh()

    def shrink(self):
        if not self.is_open:
            self.open()
        self.resize((config.track_pane_height_shrinked, config.track_pane_width))
        self.is_shrunk = True

    def magnify(self):
        if not self.is_open:
            self.open()
        self.resize((config.track_pane_height_opened, config.track_pane_width))
        self.is_shrunk = False

    def collapse(self):
        if self.is_open:
            self.track_pane.remove(self.canvas)
            self.is_open = False

    def open(self):
        if not self.is_open:
            self.track_pane.add(self.canvas)
            self.is_open = True

    def get_current_brush(self):
        return self.canvas.get_current_brush()

    def resize(self, new_dims):
        self.canvas.resize(new_dims)

    def on_resize(self, source, track):
        if self.is_shrunk:
            self.magnify()
        else:
            self.shrink()
        self.show_all()

    def on_open_or_close(self, source, track):
        if self.is_open:
            self.collapse()
        else:
            self.open()
        self.show_all()


class _TrackInfoBar(Gtk.Box, gtk_utils._IdleObject):

    __gsignals__ = {"reset_track": (GObject.SignalFlags.RUN_LAST, None, [GObject.TYPE_PYOBJECT]),
                    "resize_track": (GObject.SignalFlags.RUN_LAST, None, [GObject.TYPE_PYOBJECT]),
                    "open_or_close_track": (GObject.SignalFlags.RUN_LAST, None, [GObject.TYPE_PYOBJECT]),
                    "remove_track": (GObject.SignalFlags.RUN_LAST, None, [GObject.TYPE_PYOBJECT])}

    def __init__(self, track, width):
        Gtk.Box.__init__(self)
        self.track = track

        track_info_bar_file = os.path.join(config.glade_dir,"track_info_bar.glade")
        builder = Gtk.Builder()
        builder.add_from_file(track_info_bar_file)
        builder.connect_signals(self)

        track_info_bar_box = builder.get_object('track_info_bar_box')
        track_info_bar = builder.get_object('track_info_bar')
        track_name_field = builder.get_object('track_name_field')
        track_instrument_name_field = builder.get_object('track_instrument_name_field')

        track_info_bar_box.remove(track_info_bar)
        self.add(track_info_bar)
        self.set_size_request(width, config.track_pane_info_bar_width)

        # track_name_entry = UpdatedEntry(track, ['name'])
        # track_name_entry.connect('changed', self.on_track_name_entry_changed)
        # track_name_entry.set_on_member_update(gtk_utils.set_text_buffer_on_update)
        # track_name_entry.props.has_frame = False

        # track_instrument_name_label = UpdatedLabel(track, ['instrument_name'])
        # track_instrument_name_label.set_on_member_update(gtk_utils.set_text_buffer_on_update)
        #
        # track_name_field.add(track_name_entry)
        # track_instrument_name_field.add(track_instrument_name_label)

    ######### UI signals ###############################################################################################

    def on_track_reset_button_button_press_event(self, widget, event):
        self.emit('reset_track', self.track)

    def on_track_size_button_button_press_event(self, widget, event):
        self.emit('resize_track', self.track)

    def on_track_close_button_button_press_event(self, widget, event):
        self.emit('open_or_close_track', self.track)

    def on_track_name_entry_changed(self, entry):
        new_name = entry.get_text()
        if new_name != '':
            self.track.name = new_name

    def on_track_remove_button_button_press_event(self, widget, event):
        self.emit('remove_track', self.track)


class TrackOverview(base_widgets.Notebook):
    def __init__(self, max_height=None, min_height=None, max_width=None, min_width=None, *args, **kwargs):
        base_widgets.Notebook.__init__(self)
        self.overviews_per_track = {}
        self.child_class = None

        self.max_height = max_height
        self.min_height = min_height
        self.max_width = max_width
        self.min_width = min_width

    def add_track(self, track, *args, **kwargs):
        child = self.child_class(track, *args, **kwargs)
        self.overviews_per_track[track] = child

        # page_title_label = UpdatedLabel(track, ['name'])
        # page_title_label.set_on_member_update(gtk_utils.set_text_buffer_on_update)

        scrolled = Gtk.ScrolledWindow()

        # scrolled.set_propagate_natural_height(True)
        # scrolled.set_propagate_natural_width(True)

        self.min_height, self.min_width = config.dims_main_player_canvas[1], config.dims_main_player_canvas[0]

        if self.max_height is not None:
            scrolled.set_max_content_height(self.max_height)
        if self.min_height is not None:
            scrolled.set_min_content_height(self.min_height)

        if self.max_width is not None:
            scrolled.set_max_content_width(self.max_width)
        if self.min_width is not None:
            scrolled.set_min_content_width(self.min_width)

        scrolled.add(child)

        child.set_valign(Gtk.Align.START)
        child.set_halign(Gtk.Align.START)

        self.append_page(scrolled, track, Gtk.Label(track.name))
        config.signal_manager.add_signal('reset_track', child)
        self.show_all()

    def set_max_content_height(self, max_height):
        self.max_height = max_height
        self.foreach(Gtk.ScrolledWindow.set_max_content_height, self.max_height)

    def set_child_class(self, cla):
        assert issubclass(cla, editors.TrackEditor)
        self.child_class = cla

    def update(self):
        for child in self.overviews_per_track.values():
            child.update()

    def remove_track(self, track):
        self.remove_page(track)
        del self.overviews_per_track[track]
        self.show_all()


class InstrumentOverview(TrackOverview, gtk_utils._IdleObject):
    def __init__(self, tracks, *args, **kwargs):
        TrackOverview.__init__(self, *args, **kwargs)
        self.set_child_class(editors.InstrumentEditor)
        for track in tracks:
            self.add_track(track, *args, **kwargs)


class BufferOverview(TrackOverview, gtk_utils._IdleObject):
    def __init__(self, tracks, buffers, sound_gen_thread, *args, **kwargs):
        TrackOverview.__init__(self, *args, **kwargs)
        self.set_child_class(editors.BufferEditor)
        for track, buff in zip(tracks, buffers):
            self.add_track(track, buff, sound_gen_thread, *args, **kwargs)


class FrequencyMapOverview(TrackOverview, gtk_utils._IdleObject):
    def __init__(self, tracks, *args, **kwargs):
        TrackOverview.__init__(self, *args, **kwargs)
        self.set_child_class(editors.FrequencyMapArrayEditor)
        for track in tracks:
            self.add_track(track, *args, **kwargs)
