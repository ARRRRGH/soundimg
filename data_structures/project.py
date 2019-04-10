import data_structures.sound as sound
import data_structures.graph as graph
import data_structures.image as image
import ui.editors as editors
import ui.gtk_utils as gtk_utils
import ui.base_widgets as base_widgets
import run_time.config as config

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import os


class Project(object):
    def __init__(self):
        self.params = ProjectParams()
        self.data = ProjectData()
        self.name = 'Project'
        self.dir = '.'

    @property
    def project_file_path(self):
        return self.get_project_path(self.dir)

    @staticmethod
    def get_project_path(dir):
        return os.path.normpath(dir + '/' + config.project_file_name)

    @staticmethod
    def is_project_dir(info, switch=True):
        try:
            path = info.uri[7:]
            res = any(fn == config.project_file_name for fn in os.listdir(path))
        except NotADirectoryError:
            return False

        return res == switch

    @staticmethod
    def contains_project_dir(info, switch=True):
        try:
            path = info.uri[7:]
            res = any(fn == config.project_file_name for root, dirs, files in os.walk(path) for fn in files)
        except TypeError:
            return False

        return res == switch


class ProjectData(object):
    def __init__(self):
        self.tracks = []

    def add_track(self, params):
        new_track_widget = NewTrackWidget(params)
        dialog = base_widgets.ResponseDialog('New Track', config.main_window, new_track_widget)

        response = dialog.run()
        dialog.destroy()

        if response == Gtk.ResponseType.OK and new_track_widget.track is not None:
            self.tracks.append(new_track_widget.track)
            return new_track_widget.track

        return None


class ProjectParams(object):
    def __init__(self):
        self.sound_gen_params = SoundGenParams(config.params.image_width)


class Track(object):
    def __init__(self, params, imag=None, dims=None, limiter=None, instrument=None, freq_map_array=None):
        # gtk_utils.GtkListener.__init__(self)

        self.image = imag
        if imag is None:
            if dims is not None:
                self.image = image.Image(dims=dims)
            else:
                raise ValueError('kwarg \'image\' or kwarg \'dims\' must be supplied')

        # self.image.connect('update', self.update_from_image)

        self.limiter = limiter
        if self.limiter is None:
            self.limiter = graph.Limiter(10, name=self)
        # self.limiter.connect('update', self.update_from_limiter)

        self.instrument = instrument
        if self.instrument is None:
            self.instrument = sound.Instrument(config.wave_tables_store['Sine'])
        # self.instrument.connect('update', self.update_from_instrument)

        # self.name = 'track'
        # self.listen_to('name')
        #
        # self.instrument_name = self.instrument.name
        # self.listen_to('instrument_name')
        #
        # self.image_name = self.image.name
        # self.listen_to('image_name')

        self.freq_map_array = freq_map_array
        if self.freq_map_array is None:
            self.freq_map_array = sound.FrequencyMapArray(self, None)

        self.name = 'Track'

    @property
    def height(self):
        return self.image.img.shape[0]

    @property
    def length(self):
        return self.image.img.shape[1]

    def update_from_instrument(self, source, member, updated_value):
        if member == 'name':
            self.instrument_name = updated_value

    def update_from_limiter(self, source, member, updated_value):
        if member == 'name':
            self.limiter_name = updated_value

    def update_from_image(self, source, member, updated_value):
        if member == 'name':
            self.image_name = updated_value

    def set_without_event(self, key, value):
        self.__dict__[key] = value

    # def __getstate__(self):
    #     return self.image, self.instrument, self.limiter, self.freq_map_array


class SoundGenParams(object):
    def __init__(self, len):
        """
        Class holding parameters relevant to the sound generation from view.

        :param view:
        """
        self._output_sample_rate = config.params.output_sample_rate
        self._len_image = len
        self._duration_in_s = 10
        self._num_samples = self._duration_in_s * self._output_sample_rate
        self._samples_per_px = self._num_samples // self._len_image
        self._time_per_px = self._duration_in_s / float(self._len_image)

        self._transition_length = self._samples_per_px // 10

        # GUI

    @property
    def duration_in_s(self):
        return self._duration_in_s

    @duration_in_s.setter
    def duration_in_s(self, value):
        self._duration_in_s = value

        self._num_samples = self._duration_in_s * self._output_sample_rate
        self._samples_per_px = self._num_samples // self._len_image

        self._time_per_px = self._duration_in_s / float(self._len_image)

    @property
    def num_samples(self):
        return self._num_samples

    @num_samples.setter
    def num_samples(self, value):
        self._num_samples = value

        self._duration_in_s = self._num_samples / float(self._output_sample_rate)
        self._time_per_px = self._duration_in_s / float(self._len_image)

        self._samples_per_px = self._num_samples // self._len_image

    @property
    def samples_per_px(self):
        return self._samples_per_px

    @samples_per_px.setter
    def samples_per_px(self, value):
        self._samples_per_px = value

        self._num_samples = self._samples_per_px * self._len_image

        self._duration_in_s = self._num_samples / float(self._output_sample_rate)
        self._time_per_px = self._duration_in_s / float(self._len_image)

    @property
    def time_per_px(self):
        return self._time_per_px

    @time_per_px.setter
    def time_per_px(self, value):
        self._time_per_px = value

        self._duration_in_s = self._time_per_px * float(self._len_image)
        self._num_samples = self._duration_in_s * float(self._output_sample_rate)

        self._samples_per_px = self._num_samples // self._len_image

    @property
    def transition_length(self):
        return self._transition_length

    @transition_length.setter
    def transition_length(self, val):
        assert val <= self.samples_per_px
        self._transition_length = val

    @property
    def output_sample_rate(self):
        return self._output_sample_rate

    @output_sample_rate.setter
    def output_sample_rate(self, value):
        pass

    @property
    def dims_image(self):
        return self._len_image

    @dims_image.setter
    def dims_image(self, value):
        pass


class NewTrackWidget(Gtk.VBox):
    def __init__(self, params):
        Gtk.VBox.__init__(self)

        self.params = params

        label = Gtk.Label('Either specify number of px in y-direction and define the frequency mapping in the '
                          'options or choose from exisiting mappings')
        stack = Gtk.Stack()

        self.add(label)
        self.add(stack)

        # manual freq_map_array option page
        self.manual_freq_mapping_option_page = Gtk.VBox()
        stack.add_named(self.manual_freq_mapping_option_page, 'Manually Specify Frequency Mapping')

        self.px_entry = Gtk.Entry()
        self.px_entry.connect('changed', self.on_px_entry_changed)
        self.manual_freq_mapping_option_page.add(self.px_entry)

        # existing freq map option page
        self.existing_freq_mapping_option_page = Gtk.VBox()
        stack.add_named(self.existing_freq_mapping_option_page, 'Choose Existing Frequency Mapping')

        self.freq_map_array_editor = None
        self.track = None

    def on_px_entry_changed(self, entry):
        px = gtk_utils.read_entry(entry, int)
        self.track = Track(self.params, dims=(px, config.params.image_width))

        if self.freq_map_array_editor is None:
            self.freq_map_array_editor = editors.FrequencyMapArrayEditor(self.track)
            self.manual_freq_mapping_option_page.add(self.freq_map_array_editor)
        else:
            self.manual_freq_mapping_option_page.remove(self.freq_map_array_editor)
            self.freq_map_array_editor = editors.FrequencyMapArrayEditor(self.track)
            self.manual_freq_mapping_option_page.add(self.freq_map_array_editor)


class StartMenu(Gtk.Dialog):
    def __init__(self, name, parent):

        self.path = None

        button_data = []
        button_data += [Gtk.STOCK_OK, Gtk.ResponseType.OK]
        button_data = tuple(button_data)

        Gtk.Dialog.__init__(self, name, parent, 0, button_data)
        self.connect('response', self.on_response)

        content_area = self.get_content_area()

        logo = Gtk.Image.new_from_file('./gui/image/logo/logo_big.png')
        content_area.pack_start(logo, False, False, 0)


        self.stack = Gtk.Stack()
        # self.stack.set_vhomogeneous(False)

        self.stack_switcher = Gtk.StackSwitcher()
        self.stack_switcher.set_stack(self.stack)
        vbox = Gtk.VBox()

        vbox.pack_start(self.stack_switcher, False, False, 5)
        self.stack_switcher.props.halign = Gtk.Align.CENTER
        vbox.pack_start(self.stack, True, True, 5)

        content_area.pack_start(vbox, True, True, 5)

        # Choose from existing projects
        self.recent_project_chooser = RecentProjectChooser()
        self.stack.add_titled(self.recent_project_chooser, 'recent_project', 'Recent Project')

        self.existing_project_chooser = ExistingProjectChooser()
        self.stack.add_titled(self.existing_project_chooser, 'existing_project', 'Existing Project')

        self.new_project_widget = NewProjectWidget()
        self.stack.add_titled(self.new_project_widget, 'new_project', 'New Project')

        self.stack.set_visible_child(self.recent_project_chooser)
        self.show_all()


    def on_response(self, dialog, response_id):
        option = self.stack.get_visible_child()
        self.path = option.project_path

        if option == self.new_project_widget:
            self.make_project_dir(self.path)

    @staticmethod
    def make_project_dir(path):
        try:
            os.mkdir(os.path.normpath(path + '/data'))
            os.mkdir(os.path.normpath(path + '/data/brushes'))
            os.mkdir(os.path.normpath(path + '/data/instruments'))
            os.mkdir(os.path.normpath(path + '/data/freq_mappings'))
            os.mkdir(os.path.normpath(path + '/data/wave_tables'))
            os.mkdir(os.path.normpath(path + '/data/modulations'))
        except Exception as e:
            dialog = Gtk.MessageDialog(config.main_window, 0, Gtk.MessageType.INFO,
                                       Gtk.ButtonsType.CANCEL,
                                       "Could not create project directory from " + str(path))
            dialog.run()
            dialog.destroy()
            print(e)


class RecentProjectChooser(Gtk.Box):
    def __init__(self):
        Gtk.Box.__init__(self)
        self.project_path = None

        self.recent_chooser = Gtk.RecentChooserWidget()
        self.recent_chooser.connect('item-activated', self.on_recent_item_activated)
        self.recent_chooser.connect('selection-changed', self.on_recent_item_activated)

        self.pack_start(self.recent_chooser, True, True, 0)

        recent_filter = Gtk.RecentFilter()
        recent_filter.add_custom(Gtk.RecentFilterFlags.URI, Project.is_project_dir, True)
        self.recent_chooser.set_filter(recent_filter)

        self.show_all()


    def on_recent_item_activated(self, *args, **kwargs):
        self.project_path = self.recent_chooser.get_current_uri()


class ExistingProjectChooser(Gtk.Box):
    def __init__(self):
        Gtk.Box.__init__(self)
        self.project_path = None

        self.folder_chooser = Gtk.FileChooserWidget()
        self.folder_chooser.set_action(Gtk.FileChooserAction.SELECT_FOLDER)

        self.folder_chooser.set_create_folders(False)

        self.folder_chooser.connect('current-folder-changed', self.on_folder_changed)

        filter = Gtk.FileFilter()
        filter.add_custom(Gtk.FileFilterFlags.URI, Project.contains_project_dir, True)
        self.folder_chooser.set_filter(filter)

        self.pack_start(self.folder_chooser, True, True, 0)
        self.show_all()

    def on_folder_changed(self, *args, **kwargs):
        self.project_path = self.folder_chooser.get_current_folder()


class NewProjectWidget(Gtk.VBox):
    def __init__(self):
        Gtk.VBox.__init__(self)

        self.project_path = None

        self.folder_chooser = Gtk.FileChooserWidget('New Project')
        self.folder_chooser.set_action(Gtk.FileChooserAction.SELECT_FOLDER)
        self.folder_chooser.set_create_folders(True)

        self.folder_chooser.connect('current-folder-changed', self.on_folder_changed)

        filter = Gtk.FileFilter()
        filter.add_custom(Gtk.FileFilterFlags.URI, Project.is_project_dir, False)
        self.folder_chooser.set_filter(filter)

        self.pack_start(self.folder_chooser, True, True, 0)
        self.show_all()

    def on_folder_changed(self, *args, **kwargs):
        self.project_path = self.folder_chooser.get_current_folder()










