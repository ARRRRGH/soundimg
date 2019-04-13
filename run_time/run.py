import run_time.config as config
import data_structures.project as project
import run_time.thread as thread
import data_structures.brush as brush
import data_structures.sound as sound
import data_structures.brush_modes as brush_modes
import data_structures.wav_tables as wav_tables
import ui.editors as editors
import ui.gtk_utils as gtk_utils
import ui.widgets as widgets

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib, Gio, GObject

import sys
import shutil
import pickle
import os
import faulthandler

MENU_XML = """
<?xml version="1.0" encoding="UTF-8"?>
<interface>
  <menu id="app-menu">
    <section>
      <attribute name="label" translatable="yes">Change label</attribute>
      <item>
        <attribute name="action">win.change_label</attribute>
        <attribute name="target">String 1</attribute>
        <attribute name="label" translatable="yes">String 1</attribute>
      </item>
      <item>
        <attribute name="action">win.change_label</attribute>
        <attribute name="target">String 2</attribute>
        <attribute name="label" translatable="yes">String 2</attribute>
      </item>
      <item>
        <attribute name="action">win.change_label</attribute>
        <attribute name="target">String 3</attribute>
        <attribute name="label" translatable="yes">String 3</attribute>
      </item>
    </section>
    <section>
      <item>
        <attribute name="action">win.maximize</attribute>
        <attribute name="label" translatable="yes">Maximize</attribute>
      </item>
    </section>
    <section>
      <item>
        <attribute name="action">app.about</attribute>
        <attribute name="label" translatable="yes">_About</attribute>
      </item>
      <item>
        <attribute name="action">app.quit</attribute>
        <attribute name="label" translatable="yes">_Quit</attribute>
        <attribute name="accel">&lt;Primary&gt;q</attribute>
    </item>
    <item>
        <attribute name="action">app.save</attribute>
        <attribute name="label" translatable="yes">_Save Project</attribute>
        <attribute name="accel">&lt;Primary&gt;S</attribute>
    </item>
    <item>
        <attribute name="action">app.export_to_wav</attribute>
        <attribute name="label" translatable="yes">_Export to wav</attribute>
        <attribute name="accel">&lt;Primary&gt;E</attribute>
    </item>
    </section>
  </menu>
</interface>
"""


class App(Gtk.Application):
    def __init__(self, *args, **kwargs):
        super(App, self).__init__(*args, application_id="org.example.myapp",
                                  flags=Gio.ApplicationFlags.HANDLES_COMMAND_LINE,
                                  **kwargs)
        try:
            shutil.rmtree(config.pycache_path)
        except:
            pass

        self.window = None
        self.start_menu = project.StartMenu('Choose Project', None)

        self.add_main_option("test", ord("t"), GLib.OptionFlags.NONE,
                             GLib.OptionArg.NONE, "Command line test", None)

    def do_startup(self):
        Gtk.Application.do_startup(self)

        self.load_css()

        while self.start_menu.path is None:
            response = self.start_menu.run()
            if response != Gtk.ResponseType.OK:
                break

        if response != Gtk.ResponseType.OK:
            self.quit()
            return

        action = Gio.SimpleAction.new("about", None)
        action.connect("activate", self.on_about)
        self.add_action(action)

        action = Gio.SimpleAction.new("quit", None)
        action.connect("activate", self.on_quit)
        self.add_action(action)

        action = Gio.SimpleAction.new("save", None)
        action.connect('activate', self.on_save_project)
        self.add_action(action)

        action = Gio.SimpleAction.new("export_to_wav", None)
        action.connect('activate', self.on_export_to_wav)
        self.add_action(action)

        builder = Gtk.Builder.new_from_string(MENU_XML, -1)
        self.set_app_menu(builder.get_object("app-menu"))

    def do_activate(self):
        # We only allow a single window and raise any existing ones
        # @fixme: multiple windows could be open

        if not self.window:
            # Windows are associated with the application
            # when the last one is closed the application shuts down
            self.window = AppWindow(project_dir=self.start_menu.path, application=self, title="Main Window")

        self.start_menu.destroy()
        self.window.show()

    def do_command_line(self, command_line):
        options = command_line.get_options_dict()

        if options.contains("test"):
            # This is printed on the main instance
            print("Test argument recieved")

        self.activate()
        return 0

    def load_css(self):
        # load customized css theme
        # cssprovider = Gtk.CssProvider()
        # cssprovider.load_from_path("./theme/Communitheme-Gray/Communitheme-Gray/gtk-3.0/gtk.css")
        # screen = Gdk.Screen.get_default()
        # stylecontext = Gtk.StyleContext()
        # stylecontext.add_provider_for_screen(screen, cssprovider,
        #                                      Gtk.STYLE_PROVIDER_PRIORITY_USER)

        pass

    def on_about(self, action, param):
        about_dialog = Gtk.AboutDialog(transient_for=self.window, modal=True)
        about_dialog.present()

    def on_quit(self, action=None, param=None):
        try:
            self.window.on_main_window_destroy(action, param)
        except AttributeError:
            pass
        self.quit()

    def on_save_project(self, action, param):
        with open(self.window.project.project_file_path, 'wb') as f:
            pickle.dump(self.window.project, f)

    def on_export_to_wav(self, source, param):
        config.player.export_to_wav()


class AppWindow(Gtk.ApplicationWindow):
    def __init__(self, project_dir=None, *args, **kwargs):
        """
        Constructs the main window.

        :param args:
        :param kwargs:
        """
        super(AppWindow, self).__init__(*args, **kwargs)

        # build main window from glade file
        main_file = os.path.join(config.glade_dir, "soundimg.glade")
        main_builder = Gtk.Builder()
        main_builder.add_from_file(main_file)
        main_builder.connect_signals(self)

        # get components
        self.window = main_builder.get_object('main_window')
        config.main_window = self.window

        # initialize_plt_objects thread manager
        config.thread_manager = thread.ThreadManager(10)

        # initialize signal manager
        config.signal_manager = gtk_utils.SignalManager()

        # set initial flags
        self.project = self.load_project(project_dir)
        config.project = self.project

        main_field = main_builder.get_object('main_field')

        player_toolbar_box = main_builder.get_object('player_toolbar_box')
        overviews_toolbar_field = main_builder.get_object('options_toolbar_box')

        brush_editor_box = main_builder.get_object('brush_editor_box')
        spinner_field = main_builder.get_object('sound_gen_spinner_field')

        amp_field = main_builder.get_object('main_amp_field')

        # initialize brush editor
        brush_window = self.set_brush_editor(brush_editor_box)
        config.brush_window = brush_window

        # set eclipse window in main_field
        table = widgets.Table(self.project, dims=config.dims_main_player_canvas)
        overviews_toolbar_field.pack_start(table.option_toolbar, False, False, 0)
        main_field.pack_start(table, False, False, 0)

        # initialize buffer view
        self.set_buffer_view(amp_field, config.player.sound_gen_thread.buffer)

        # move toolbar outside main_player
        gtk_utils.move_widget_btw_boxes(config.player.player_toolbar, config.player.player_grid, player_toolbar_box)

        # connect spinner
        spinner_field.add(widgets.SpinnerOnSignal(config.player.sound_gen_thread, 'not_idle', 'idle'))

    def set_buffer_view(self, parent, buff):
        buffer_graph_plot = editors.BufferView(buff=buff, dims=config.dims_amp_view, orientation=1, direction_x=-1)
        buffer_graph_plot.toolbar.hide_message(True)
        parent.pack_start(buffer_graph_plot, False, False, 0)

    def set_brush_editor(self, parent):
        """
        Constructs a BrushEditor on imag from brush_editor_glade_file and adds it to menu_button.

        :param parent: Gtk.Container which will hold the brush editor
        :param imag:
        :param brush_editor_glade_file:
        :param menu_button:
        :return:
        """
        brush_editor = editors.BrushWindow(dims_canvas=config.dims_brush_preview_canvas)
        config.signal_manager.add_signal('set-new-current-brush', brush_editor)
        parent.pack_start(brush_editor, False, False, 0)

        return brush_editor

    def show(self):
        self.window.show_all()

    def load_project(self, project_dir=None):
        """
        Load saved files and data from previous sessions.

        :return:
        """
        project_loader_file_path = project.Project.get_project_path(project_dir)

        if os.path.exists(project_loader_file_path):
            with open(project_loader_file_path, 'rb') as f:
                proj = pickle.load(f)
        else:
            proj = project.Project()

        config.project = proj

        proj.dir = project_dir

        self.load_brush_modes()
        self.load_brushes(project_dir)
        self.load_wave_tables(project_dir)
        self.load_instruments(project_dir)
        self.load_freq_mappings(project_dir)
        self.load_modulations(project_dir)

        return proj

    def load_brush_modes(self):
        brush_mode_store = gtk_utils.FileStore(storage_dir=None, ext=None, typ=brush_modes.BrushMode)
        config.brush_mode_store = brush_mode_store

        config.brush_mode_store.add_to_store(brush_modes.AddMode(), 'Add', protected=True)
        config.brush_mode_store.add_to_store(brush_modes.AdditiveMixingMode(), 'Additive Mixing', protected=True)
        config.brush_mode_store.add_to_store(brush_modes.SubtractMode(), 'Subtract', protected=True)
        config.brush_mode_store.add_to_store(brush_modes.NegativeMixingMode(), 'Negative Mixing', protected=True)
        config.brush_mode_store.add_to_store(brush_modes.FilterMode(), 'Filter', protected=True)

    def load_brushes(self, proj_dir):
        # add basic brushes
        brush_store = gtk_utils.FileStore(storage_dir=os.path.join(proj_dir, 'data/brushes'), ext='brush',
                                          typ=brush.Brush)
        config.brush_store = brush_store

        add_mode = config.brush_mode_store['Add']
        config.brush_store.add_to_store(brush.Line(add_mode), 'Line', protected=True)
        config.brush_store.add_to_store(brush.InfiniteLine(add_mode), 'Infinite Line', protected=True)
        config.brush_store.add_to_store(brush.Rectangle(add_mode), 'Rectangle', protected=True)

    def load_instruments(self, proj_dir):
        instrument_store = gtk_utils.FileStore(storage_dir=os.path.join(proj_dir, 'data/instruments'),
                                               ext='instrument', typ=sound.Instrument)
        config.instrument_store = instrument_store

        sine = config.wave_tables_store['Sine']
        config.instrument_store.add_to_store(sound.Instrument(sine), 'Basic Sine', protected=True)


    def load_wave_tables(self, proj_dir):
        wave_tables_store = gtk_utils.FileStore(storage_dir=os.path.join(proj_dir, 'data/wave_tables'),
                                                ext='wavetable', typ=wav_tables.WaveTable)
        config.wave_tables_store = wave_tables_store

        sine = wav_tables.Sine()
        config.wave_tables_store.add_to_store(sine, sine.name, protected=True)

        noise = wav_tables.WhiteNoise()
        config.wave_tables_store.add_to_store(noise, noise.name, protected=True)


    def load_freq_mappings(self, proj_dir):
        freq_mappings_store = gtk_utils.FileStore(storage_dir=os.path.join(proj_dir, 'data/freq_mappings'),
                                                  ext='freq_mapping', typ=sound.FrequencyMapArray)
        config.freq_mappings_store = freq_mappings_store


    def load_modulations(self, proj_dir):
        modulation_store = gtk_utils.FileStore(storage_dir=os.path.join(proj_dir, 'data/modulations'),
                                               ext='modulation', typ=sound.Timbre)
        config.modulation_store = modulation_store


    ######### UI signals ###############################################################################################

    def on_brush_editor_button_clicked(self, source):
        if self.brush_editor.get_visible():
            self.brush_editor.hide()
        else:
            self.brush_editor.show_all()

    def on_main_window_destroy(self, sender=None, data=None):
        """
        Quit program after killing all threads

        :param sender:
        :param data:
        :return:
        """
        config.player.pause(regenerate=False)
        config.thread_manager.stop_all_threads(block=True)
        self.window.close()

        try:
            shutil.rmtree(config.pycache_path)
        except:
            pass


def run():
    faulthandler.enable()
    GObject.threads_init()
    app = App()
    app.run(sys.argv)


if __name__ == "__main__":
    run()
