import run_time.py_utils as py_utils
import data_structures.image as image
import data_structures.brush as brush
import ui.base_widgets as base_widgets
import ui.gtk_utils as gtk_utils

import gi

from data_structures import graph as graph
from run_time import config as config
from ui import imaging as imaging

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GObject, Gdk

import os
import numpy as np
import copy
from abc import abstractmethod


class TrackEditor(Gtk.VBox):
    def __init__(self):
        Gtk.VBox.__init__(self)

        toolbar = Gtk.Toolbar()
        self.add(toolbar)

        apply_button = imaging.NavigationToolButton('Apply', 'Apply Settings', 'reload', self.update)
        save_button = imaging.NavigationToolButton('Save', 'Save Instrument', 'save', self.save)

        toolbar.insert(apply_button, -1)
        toolbar.insert(save_button, -1)


    @abstractmethod
    def update(self):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError


class GraphEditor(Gtk.VBox):
    """
    This editor shows the image.Graph objects added to it and allows to add, move and remove data points of the
    active graph interactively. In order to swap between moving and removing points, remove_mode must be set.

    By default, a toggle button is added to the content area that allows toggling between remove_mode states.

    Graphs can be added to the editor by calling set_new_graph(graph, draw_graph_callable) where draw_graph_callable
    is the plotting function called upon each refresh of the editor. For information on the exact use of draw_graph_callable,
    consult, UpdatedGraphPlot.set_new_graph. The active graph can be changed by calling set_active_graph(new_active_graph).

    To use this editor, at least the following is required after instantiation.
        1. Add an image.Graph object
        2. Set mouse sensibility in plot units
        3. refresh

    """

    def __init__(self, graph_plot=None, remove_mode=False, *args, **kwargs):
        """
        :param args: for UpdatedGraphPlot instantiation
        :param kwargs: for UpdatedGraphPlot instantiation
        """
        Gtk.VBox.__init__(self)

        self.graph_plot = graph_plot
        if graph_plot is None:
            self.graph_plot = imaging.UpdatedGraphPlot(*args, **kwargs)
        self.pack_start(self.graph_plot, False, False, 0)

        # initially toolbar's active graph points to None, must wait for set_active_graph
        self.graph_plot.content_area.remove(self.graph_plot.toolbar)
        self.graph_plot.toolbar = imaging.GraphNavigationToolbar(self, None, config.main_window)
        self.graph_plot.content_area.pack_start(self.graph_plot.toolbar, True, True, 0)

        self.remove_mode = remove_mode

        self.active_graph = None
        self.active_data_point_ind = None

        self.press_data_point_threshold_x = None
        self.press_data_point_threshold_y = None

        self.current_action = None
        self.action_data = []

        self.apply_insert_point_rules = None

        def return_same(coord_data, **kwargs):
            return coord_data

        self.set_insert_point_rules(return_same)

    @property
    def graphs(self):
        return self.graph_plot.graphs

    @property
    def content_area(self):
        return self.graph_plot.content_area

    def refresh(self, *args, **kwargs):
        """
        Refreshes all graphs shown by this editor.

        :param args:
        :param kwargs:
        :return:
        """
        self.graph_plot.refresh(*args, **kwargs)

    def set_new_graph(self, graph, draw_graph_callable, update_graph_callable):
        """

        :param graph: graph object to be added to this editor
        :param draw_graph_callable: plotting function called on refresh of graph. For exact use, look up
                                    UpdatedGraphImage.set_new_graph
        :return:
        """
        self.graph_plot.set_new(graph, draw_graph_callable, update_graph_callable)
        if len(self.graphs) == 1:
            self.set_active_graph(graph)

    def set_mouse_sensibility(self, in_y_dir, in_x_dir):
        """
        Set mouse sensibility in plot units.
        :param in_y_dir:
        :param in_x_dir:
        :return:
        """
        self.press_data_point_threshold_x = in_x_dir
        self.press_data_point_threshold_y = in_y_dir

    def set_active_graph(self, graph):
        if graph in self.graphs:
            self.active_graph = graph
            self.graph_plot.toolbar.graph = graph
        else:
            raise ValueError('Graph must be added to editor first. Call set_new_graph first.')

    def set_insert_point_rules(self, rules_callable):
        """
        Apply rules on coordinates when new point is inserted, e.g. to exclude a specific range.
        :param rules_callable: A callable that takes the tuple (y_coord, x_coord) and returns (new_y_coord, new_x_coord)
        :return:
        """
        self.apply_insert_point_rules = rules_callable

    def set_width(self, *args, **kwargs):
        """
        Sets width of editor in plot units. Normally this is the extent of the longest graph. Must only be set when
        orientation_x == -1
        :param args:
        :param kwargs:
        :return:
        """
        self.graph_plot.set_x_mirr(*args, **kwargs)

    def set_remove_mode(self, mode):
        """
        :param mode: boolean deciding whether clicked points are moved or removed
        :return:
        """
        raise NotImplementedError


class TransitionEditor(Gtk.VBox):
    def __init__(self, instrument):
        Gtk.VBox.__init__(self)

        self.instrument = instrument
        self.fader_in = self.instrument.fader_in
        self.fader_out = self.instrument.fader_out

        fader_in_editor = FadeEditor(self.fader_in, dims=config.dims_fader_editor)
        fader_out_editor = FadeEditor(self.fader_out, dims=config.dims_fader_editor)

        fade_editor_area = Gtk.Grid()
        fade_editor_area.set_row_spacing(config.default_rowspacing)
        fade_editor_area.set_column_spacing(config.default_colspacing)
        self.add(fade_editor_area)

        fade_opts_area = Gtk.HBox()
        self.add(fade_opts_area)

        fade_editor_area.attach(fader_in_editor, 0, 0, 1, 1)
        fade_editor_area.attach(fader_out_editor, 1, 0, 1, 1)

    def get_fade_in(self):
        return self.fader_in.get()[0]

    def get_fade_out(self):
        return self.fader_out.get()[0]

    def update(self):
        self.instrument.update_fade_arrs()


class FadeEditor(GraphEditor):
    def __init__(self, fader_graph, *args, **kwargs):
        self.fader_graph = fader_graph

        self.fader_plot = imaging.UpdatedGraphPlot(dims=config.dims_fader_editor)
        self.fader_plot.set_new(fader_graph, self.draw_fader, self.update_fader)

        GraphEditor.__init__(self, self.fader_plot, *args, **kwargs)

        # editor settings
        self.set_mouse_sensibility(0.1, 0.05)
        self.set_insert_point_rules(self.insert_point_rules)
        self.set_active_graph(fader_graph)

        settings = [('Transition Length', 'transition_length', int, 'entry',
                     {'range': (1, config.project.params.sound_gen_params.samples_per_px), 'increments': (1,1)}, 8, 1)]
        self.settings_dialog = base_widgets.SettingsDialog(settings, self, 'Set Transition Length', config.main_window)

        settings_button = imaging.NavigationToolButton('...', 'Settings', '...', self.show_settings_dialog)
        self.fader_plot.toolbar.add_tool(settings_button)

    @property
    def transition_length(self):
        return self.fader_graph.width

    @transition_length.setter
    def transition_length(self, val):
        self.fader_graph.width = val
        self.fader_plot.redraw()

    def show_settings_dialog(self, source):
        self.settings_dialog.run()

    def draw_fader(self, ax, data, **kwargs):
        coords, coords_smoothed = data
        ys, xs = coords
        ys_smoothed, xs_smoothed = coords_smoothed

        collection = ax.scatter(xs, ys)
        line, = ax.step(xs_smoothed, ys_smoothed)

        ax.set_ylim([-0.1, 1.1])
        # ax.set_xlim([-.1,1.1])

        # ax.axis('off')
        return collection, line

    def update_fader(self, ax, artists, data):
        coords, coords_smoothed = data
        ys, xs = coords
        offsets = np.array([xs, ys]).T
        ys_smoothed, xs_smoothed = coords_smoothed
        collection, line = artists

        collection.set_offsets(offsets)
        line.set_ydata(ys_smoothed)

        return collection, line

    def insert_point_rules(self, coord_data, *args, **kwargs):
        y, x = coord_data
        return max(0, y), min(max(x, 0), 1)


class LimiterEditor(GraphEditor):
    """
    GraphEditor consisting of a background graph and a limiter.
    """

    def __init__(self, buff, limiter_graph, *args, **kwargs):
        graph_plot = BufferView(buff, *args, **kwargs)
        graph_plot.set_new(limiter_graph, self.draw_limiter, self.update_limiter)

        GraphEditor.__init__(self, graph_plot, *args, **kwargs)

        # editor settings
        self.set_mouse_sensibility(0.1, 0.05)
        self.set_insert_point_rules(self.insert_point_rules)
        self.set_active_graph(limiter_graph)

        # add a time pointer as an overlay to graph_plot
        self.time_pointer = imaging.TimePointer(self.graph_plot, config.player)

        self.refresh(source=None)

    def insert_point_rules(self, coord_data, *args, **kwargs):
        y, x = coord_data
        return max(0, y), min(max(x, 0), 1)

    def draw_limiter(self, ax, data, **kwargs):
        coords, coords_smoothed = data
        ys, xs = coords
        ys_smoothed, xs_smoothed = coords_smoothed

        collection = ax.scatter(xs, ys)
        line1, = ax.step(xs_smoothed, ys_smoothed)
        line2, = ax.step(xs_smoothed, -ys_smoothed)

        ax.set_ylim([-1.1, 1.1])
        # ax.set_xlim([-.1,1.1])

        ax.axis('off')
        return collection, line1, line2

    def update_limiter(self, ax, artists, data):
        coords, coords_smoothed = data
        ys, xs = coords
        offsets = np.array([xs, ys]).T
        ys_smoothed, xs_smoothed = coords_smoothed
        collection, line1, line2 = artists

        collection.set_offsets(offsets)
        line1.set_ydata(ys_smoothed)
        line2.set_ydata(-ys_smoothed)

        return collection, line1, line2


class _ApplyBrushMenu(Gtk.Box):

    __gsignals__ = {'changed': (GObject.SignalFlags.RUN_LAST, None, [])}

    def __init__(self, canvas):
        Gtk.Box.__init__(self, orientation=Gtk.Orientation.VERTICAL)

        self.canvas = canvas

        # define brush application with size scale and brush store
        self.brush_combo = base_widgets.StoreComboBox(config.brush_store)
        self.brush_combo.connect('changed', self.on_brush_changed)

        self.size_scale = Gtk.Scale()
        self.size_scale.connect('value-changed', self.on_size_scale_value_changed)

        self.size_scale.set_range(1, config.params.image_width)
        self.size_scale.set_increments(1, 1)
        self.size_scale.set_digits(0)
        self.size_scale.set_value_pos(Gtk.PositionType.RIGHT)

        # define the repeater
        self.repeater_menu = _RepeaterMenu()
        self.repeater_menu.connect('changed', self.on_repeater_changed)

        # define color selection
        self.colmode_menu = _ColorAndModeMenu()
        self.colmode_menu.connect('changed', self.on_colmode_changed)

        # attach objects to grid
        main_box = Gtk.Grid()
        main_box.set_row_spacing(config.default_rowspacing)
        main_box.set_column_spacing(config.default_colspacing)
        self.add(main_box)

        # br_box = Gtk.HBox()
        main_box.attach(Gtk.Label('Apply Brush'), 0, 0, 1, 1)
        main_box.attach(self.size_scale, 1, 0, 3, 1)
        main_box.attach(self.brush_combo, 1, 1, 2, 1)

        # br_box.pack_start(self.size_scale, True, True, 0)
        # br_box.pack_end(self.brush_combo, False, False, 0)

        main_box.attach(Gtk.Label('Repeat Brush'), 0, 2, 1, 1)
        main_box.attach(self.repeater_menu, 1, 2, 3, 1)

        main_box.attach(Gtk.Label('Color'), 0, 3, 1, 1)
        main_box.attach(self.colmode_menu, 1, 3, 3, 2)

        self.show_all()

    def on_size_scale_value_changed(self, sender=None, data=None, data2=None):
        self.update_brush_from_size_scale(self.active_brush, self.size_scale)

    def on_brush_changed(self, combo=None, data=None):
        self.update_brush_from_size_scale(self.active_brush, self.size_scale)
        self.emit('changed')

    def update_brush_from_size_scale(self, brush, scale):
        if brush is not None:
            brush.set_shape_from_scalar(scale.get_value())
            brush.update(self.canvas.imag)

    def update_brush_color(self):
        try:
            self.active_brush.set_color(self.colmode_menu.rgb)
        except Exception as e:
            print(e)

    def update_brush_mode(self):
        self.active_brush.set_mode(self.colmode_menu.mode)

    def on_repeater_changed(self, *args, **kwargs):
        self.emit('changed')

    def on_colmode_changed(self, *args, **kwargs):
        self.update_brush_color()
        self.update_brush_mode()
        self.update_weight()
        self.emit('changed')

    def update_weight(self):
        self.active_brush.set_weights(self.colmode_menu.weight)

    @property
    def active_brush(self):
        return self.brush_combo.get_active_value()


class _ColorAndModeMenu(Gtk.HBox):

    __gsignals__ = {'changed': (GObject.SignalFlags.RUN_LAST, None, [])}

    def __init__(self):
        Gtk.HBox.__init__(self)

        self.content_area = Gtk.VBox()

        self.pack_start(self.content_area, True, True, 0)
        self.rgb = None
        self.mode = None
        self.weight = None

        self.color_button = Gtk.ColorButton()
        self.color_button.connect('color-set', self.on_color_changed)

        self.mode_combo = base_widgets.StoreComboBox(config.brush_mode_store, info_button=True)
        self.mode_combo.hide_import_button()
        self.mode_combo.connect('changed', self.on_mode_changed)

        self.weight_scale = Gtk.Scale()
        self.weight_scale.connect('value-changed', self.on_weight_scale_value_changed)
        self.weight_scale.set_range(0, 1)
        self.weight_scale.set_increments(0.01, 0.1)
        self.weight_scale.set_digits(2)
        self.weight_scale.set_value_pos(Gtk.PositionType.RIGHT)

        self.cm_content_area = Gtk.HBox()
        self.content_area.pack_start(self.cm_content_area, True, True, 0)

        self.cm_content_area.pack_start(self.color_button, False, False, 0)
        self.cm_content_area.pack_start(self.mode_combo, False, False, config.default_hborder)

        self.content_area.pack_start(self.weight_scale, True, True, 0)

        self.show_all()

    def on_weight_scale_value_changed(self, sender=None, data=None, data2=None):
        self.weight = self.weight_scale.get_value()
        self.emit('changed')

    def on_color_changed(self, source):
        rgba = self.color_button.get_rgba()
        color = rgba.to_color()
        hex_s = color.to_string()

        self.rgb = py_utils.hex_to_rgb(hex_s)
        self.emit('changed')

    def on_mode_changed(self, source):
        self.mode = self.mode_combo.get_active_value()
        self.emit('changed')


class _RepeaterMenu(Gtk.HBox):

    __gsignals__ = {'changed': (GObject.SignalFlags.RUN_LAST, None, [])}

    def __init__(self):
        Gtk.HBox.__init__(self)

        self.spacing_entry = Gtk.Entry()
        self.repetitions_entry = Gtk.Entry()
        self.toggle_x_repeat = Gtk.CheckButton.new_with_label('x')
        self.toggle_y_repeat = Gtk.CheckButton.new_with_label('y')

        self.pack_start(self.spacing_entry, False, False, 0)
        self.pack_start(self.repetitions_entry, True, False, config.default_hborder)
        self.pack_start(self.toggle_x_repeat, True, False, 0)
        self.pack_end(self.toggle_y_repeat, False, False, 0)

        self.spacing_entry.set_width_chars(10)
        self.spacing_entry.set_text('Spacing')

        self.repetitions_entry.set_width_chars(10)
        self.repetitions_entry.set_text('Repetitions')

        self.spacing_entry.connect('changed', self.on_repeater_options_changed)
        self.repetitions_entry.connect('changed', self.on_repeater_options_changed)
        self.toggle_x_repeat.connect('toggled', self.on_repeat_toggled)
        self.toggle_y_repeat.connect('toggled', self.on_repeat_toggled)

    @property
    def spacing(self):
        return gtk_utils.read_entry(self.spacing_entry, int)

    @property
    def repetitions(self):
        return gtk_utils.read_entry(self.repetitions_entry, int)

    def on_repeater_options_changed(self, *args, **kwargs):
        self.emit('changed')

    def on_repeat_toggled(self, *args, **kwargs):
        self.emit('changed')


class _BrushEditor(Gtk.VBox):

    __gsignals__ = {'changed': (GObject.SignalFlags.RUN_LAST, None, [])}

    def __init__(self, dims_imag, dims_canvas):
        Gtk.VBox.__init__(self)

        imag = image.Image(dims=dims_imag)

        if dims_canvas is None:
            dims_canvas = dims_imag

        upper_toolbar = Gtk.Toolbar()
        self.add(upper_toolbar)

        # set up canvas
        self.canvas = imaging.Canvas(dims_canvas=dims_canvas, imag=imag)
        self.add(self.canvas)

        # set up init brush opts
        brush_editor_effects_box = Gtk.HBox()
        brush_editor_effects_box.set_border_width(5)
        self.add(brush_editor_effects_box)

        # init brush combo
        new_from_existing_box = Gtk.VBox()
        new_from_existing_box.set_border_width(5)
        new_from_existing_box.props.width_request = 200

        self.brush_combo = base_widgets.StoreComboBox(config.brush_store)
        new_from_existing_box.pack_start(self.brush_combo, False, False, 0)

        brush_tool = Gtk.ToolItem()
        menu_button = Gtk.MenuButton('Copy from Brush')

        popover = Gtk.Popover()
        popover.add(new_from_existing_box)
        popover.set_transitions_enabled(False)

        menu_button.set_popover(popover)
        brush_tool.add(menu_button)
        upper_toolbar.add(brush_tool)

        # init brush x and y entry
        self.brush_editor_x_dim_entry = Gtk.Entry()
        self.brush_editor_y_dim_entry = Gtk.Entry()

        self.brush_editor_x_dim_entry.connect('activate', self.on_x_dim_entry_activate)
        self.brush_editor_y_dim_entry.connect('activate', self.on_y_dim_entry_activate)

        self.brush_editor_x_dim_entry.connect('changed', self.on_x_dim_entry_changed)
        self.brush_editor_y_dim_entry.connect('changed', self.on_y_dim_entry_changed)

        self.brush_editor_x_dim_entry.set_width_chars(3)
        self.brush_editor_x_dim_entry.set_text(str(self.canvas.imag.shape[1]))

        self.brush_editor_y_dim_entry.set_width_chars(3)
        self.brush_editor_y_dim_entry.set_text(str(self.canvas.imag.shape[0]))

        xy_box = Gtk.HBox()
        xy_box.add(Gtk.Label('x-dim'))
        xy_box.pack_start(self.brush_editor_x_dim_entry, False, False, config.default_hborder)
        xy_box.add(Gtk.Label('y-dim'))
        xy_box.pack_start(self.brush_editor_y_dim_entry, False, False, config.default_hborder)

        xy_box.set_border_width(5)

        new_from_dims_tool = Gtk.ToolItem()
        menu_button = Gtk.MenuButton('New Empty')

        popover2 = Gtk.Popover()
        popover2.add(xy_box)
        popover2.set_transitions_enabled(False)
        menu_button.set_popover(popover2)

        new_from_dims_tool.add(menu_button)
        upper_toolbar.add(brush_tool)
        upper_toolbar.add(new_from_dims_tool)

        # apply button
        apply_button = imaging.NavigationToolButton('Apply', 'Apply Settings', 'reload',
                                                    self.on_apply_initial_button_clicked)
        save_button = imaging.NavigationToolButton('Save', 'Save editor brush', 'save',
                                                   self.on_store_menu_button_clicked)

        upper_toolbar.insert(apply_button, -1)
        upper_toolbar.insert(save_button, -1)

        # set up scales
        self.size_scale = Gtk.Scale()
        self.brush_editor_blur_scale = Gtk.Scale()

        self.size_scale.connect('value-changed', self.on_init_size_scale_value_changed)
        self.brush_editor_blur_scale.connect('value-changed', self.on_blur_scale_value_changed)

        new_from_existing_box.add(self.size_scale)
        self.add(self.brush_editor_blur_scale)

        self.size_scale.set_range(1, 2)
        self.size_scale.set_increments(0.1, 0.01)

        self.brush_editor_blur_scale.set_range(1, 2)
        self.brush_editor_blur_scale.set_increments(0.1, 0.01)

        # add editor preview to brush_store as a brush
        editor_brush = brush.EditorBrush(self.canvas.imag.img, np.ones(self.canvas.imag.shape))
        config.brush_store.add_to_store(editor_brush, '<editor>', protected=False)

        self.update_brush_scale_ranges()

        self.show_all()
        popover.show_all()
        popover.popdown()
        popover2.show_all()
        popover2.popdown()

    def on_apply_initial_button_clicked(self, sender, data=None):
        shape = self.get_shape_from_entries()
        if shape[0] is None or shape[1] is None:
            return

        # reset cavas image
        self.canvas.reset(*shape)

        active_brush = self.brush_combo.get_active_value()
        if active_brush is not None:
            center = (self.canvas.imag.shape[0] // 2, self.canvas.imag.shape[1] // 2)
            active_brush.set_shape_from_scalar(self.size_scale.get_value())
            active_brush.apply(center, self.canvas.imag)

        # reset <editor> brush
        new_editor_brush = brush.EditorBrush(self.canvas.imag.img, np.ones(self.canvas.imag.shape))
        config.brush_store.replace(config.brush_store['<editor>'], new_editor_brush)

        self.canvas.redraw()
        self.canvas.toolbar.update()

    def get_shape_from_entries(self):
        try:
            x_dim = gtk_utils.read_entry(self.brush_editor_x_dim_entry)
        except:
            self.brush_editor_x_dim_entry.set_text('')
            x_dim = None

        try:
            y_dim = gtk_utils.read_entry(self.brush_editor_y_dim_entry)
        except:
            self.brush_editor_y_dim_entry.set_text('')
            y_dim = None

        return y_dim, x_dim

    def try_to_pad(self, shape):
        shape = list(shape)

        if shape[0] is None:
            shape[0] = self.canvas.imag.shape[0]
        if shape[1] is None:
            shape[1] = self.canvas.imag.shape[1]

        shape = tuple(shape)
        self.canvas.extend(shape)
        self.canvas.refresh()

    def update_brush_from_size_scale(self, brush, scale):
        if brush is not None:
            brush.set_shape_from_scalar(scale.get_value())
            brush.update(self.canvas.imag)

    def on_store_menu_button_clicked(self, source):
        menu = base_widgets.StoreDialog(config.brush_store, key='<editor>', action=Gtk.FileChooserAction.SAVE)
        response = menu.run()

    def on_init_size_scale_value_changed(self, scale, data=None, data2=None):
        self.update_brush_from_size_scale(self.brush_combo.get_active_value(), self.size_scale)

    def on_blur_scale_value_changed(self, source):
        scale_value = int(self.brush_editor_blur_scale.get_value())

        self.canvas.imag.img[self.canvas.current_selection] = py_utils.ImageHandler.blur(self.canvas.imag.img, scale_value)
        self.canvas.refresh()

    def on_y_dim_entry_changed(self, source):
        self.update_brush_scale_ranges()

    def on_x_dim_entry_changed(self, source):
        self.update_brush_scale_ranges()

    def on_x_dim_entry_activate(self, source):
        shape = self.get_shape_from_entries()
        self.try_to_pad(shape)

    def on_y_dim_entry_activate(self, source):
        shape = self.get_shape_from_entries()
        self.try_to_pad(shape)

    def update_brush_scale_ranges(self):
        try:
            shape = self.get_shape_from_entries()
            y_dim, x_dim = shape

            if y_dim is None or x_dim is None:
                return

            self.size_scale.set_range(1, max(x_dim, y_dim))
            self.brush_editor_blur_scale.set_range(1, max(x_dim, y_dim) // 2)
        except:
            pass


class DrawingWindow(Gtk.Box):

    __gsignals__ = {'set-new-current-brush': (GObject.SignalFlags.RUN_LAST, None, [GObject.TYPE_PYOBJECT])}

    def __init__(self, dims_canvas=None, dims_imag=None):
        Gtk.Box.__init__(self)

        scrolled = base_widgets.CollapsingSectionWindow(propagate_natural_height=False, propagate_natural_width=False,
                                                        min_dims=config.dims_brush_window)
        self.pack_start(scrolled, True, True, 0)

        self.brush_editor = _BrushEditor(dims_canvas=dims_canvas, dims_imag=dims_imag)
        self.brush_editor.connect('changed', self.on_brush_editor_changed)

        self.apply_menu = _ApplyBrushMenu(self.brush_editor.canvas)
        self.apply_menu.connect('changed', self.on_apply_menu_changed)

        scrolled.add_child(self.apply_menu, border=config.section_border,
                           rubberband=config.section_rubberband, title_label=Gtk.Label('Brush'))

        scrolled.add_child(self.brush_editor, border=config.section_border,
                           rubberband=config.section_rubberband, title_label=Gtk.Label('Brush Editor'))

    @property
    def active_brush_is_x_repeated(self):
        return self.apply_menu.repeater_menu.toggle_x_repeat.get_active()

    @property
    def active_brush_is_y_repeated(self):
        return self.apply_menu.repeater_menu.toggle_y_repeat.get_active()

    @property
    def current_brush(self):
        return self.brush_editor.canvas.current_brush

    @current_brush.setter
    def current_brush(self, val):
        self.brush_editor.canvas.current_brush = val
        self.emit('set-new-current-brush', self.brush_editor.canvas.current_brush)

    def set_current_brush(self):
        active_brush = self.apply_menu.active_brush
        if active_brush is not None and (self.active_brush_is_x_repeated or self.active_brush_is_y_repeated):
            try:
                spacing = self.apply_menu.repeater_menu.spacing
                repetitions = self.apply_menu.repeater_menu.repetitions

                self.current_brush = brush.Repeater(active_brush,
                                                    spacing=spacing,
                                                    nr_repetitions=repetitions,
                                                    x_axis_repeated=self.active_brush_is_x_repeated,
                                                    y_axis_repeated=self.active_brush_is_y_repeated)
            except TypeError:
                pass
        else:
            self.current_brush = active_brush

    def update_current_brush(self):
        self.set_current_brush()

    def on_apply_menu_changed(self, *args, **kwargs):
        self.update_current_brush()

    def on_brush_editor_changed(self, *args, **kwargs):
        self.update_current_brush()


class BufferEditor(TrackEditor):
    __gsignals__ = {"reset_track": (GObject.SignalFlags.RUN_LAST, None, [GObject.TYPE_PYOBJECT])}

    def __init__(self, track, buff, sound_gen_thread, *args, **kwargs):
        TrackEditor.__init__(self)

        buffer_editor_file = os.path.join(config.glade_dir, "buffer_editor.glade")
        builder = Gtk.Builder()
        builder.add_from_file(buffer_editor_file)

        # get objects
        self.buffer_editor_window = builder.get_object('buffer_editor_window')
        self.buffer_editor = builder.get_object('buffer_editor')
        self.buffer_preview_field = builder.get_object('buffer_preview')

        # connect builder objects to this class
        builder.connect_signals(self)
        self.buffer_editor_window.remove(self.buffer_editor)
        self.pack_start(self.buffer_editor, False, False, 0)

        self.limiter_editor = LimiterEditor(buff, track.limiter, dims=config.dims_track_limiter_editor)

        sound_gen_thread.connect('idle', self.limiter_editor.refresh)
        self.buffer_preview_field.add(self.limiter_editor)

        self.props.valign = Gtk.Align.FILL

    def update(self):
        self.emit('reset_track', self.track)


class InstrumentEditor(TrackEditor):
    __gsignals__ = {"reset_track": (GObject.SignalFlags.RUN_LAST, None, [GObject.TYPE_PYOBJECT])}

    def __init__(self, track, wt=None, min_dims=None, *args, **kwargs):
        TrackEditor.__init__(self)

        self.track = track
        self.instrument = track.instrument

        if wt is None:
            self.wave_table = track.instrument.wave_table
        else:
            self.wave_table = wt

        coll_window = base_widgets.CollapsingSectionWindow(min_dims=min_dims, *args, **kwargs)
        self.add(coll_window)

        # set wave table editor
        self.wave_editor = WaveTableEditor(self.instrument)

        # set modulation editor
        self.modulation_table = ModulationTable(self.instrument)

        # set fader
        self.transition_editor = TransitionEditor(self.instrument)

        coll_window.add_child(self.wave_editor, border=config.section_border, hborder=config.section_hborder,
                              rubberband=config.section_rubberband, title_label=Gtk.Label('Wave Table'))

        coll_window.add_child(self.modulation_table, border=config.section_border, hborder=config.section_hborder,
                              rubberband=config.section_rubberband, title_label=Gtk.Label('Modulation'))

        coll_window.add_child(self.transition_editor, border=config.section_border, hborder=config.section_hborder,
                              rubberband=config.section_rubberband, title_label=Gtk.Label('Fade In / Fade Out'))

        self.show()
        self.update()

    ######### UI signals ###############################################################################################

    def update(self, source=None):
        self.wave_editor.update()
        self.modulation_table.update()
        self.transition_editor.update()
        self.emit('reset_track', self.track)

    def save(self, source=None):
        menu = base_widgets.StoreDialog(config.instrument_store, item=self.instrument,
                                        action=Gtk.FileChooserAction.SAVE)
        response = menu.run()


class WaveTableEditor(Gtk.VBox):
    def __init__(self, instrument):
        Gtk.VBox.__init__(self)
        self.instrument = instrument

        self.notebook = Gtk.Notebook()
        self.notebook.set_show_border(False)
        self.initialize_notebook()

        self.add(self.notebook)

        self.wave_table = self.instrument.wave_table
        self.preview = imaging.WaveTablePreview(self.wave_table, dims=config.dims_wave_table_preview)
        self.update()

        self.add(self.preview)

        insert_button = imaging.NavigationToolButton('Add', 'Add other wave table', 'add', self.add_constituent)
        remove_button = imaging.NavigationToolButton('Remove', 'Remove constituent', 'remove', self.remove_constituent)
        save_button = imaging.NavigationToolButton('Save', 'Save Wave Table', 'save', self.save)

        self.preview.toolbar.add_tool(insert_button)
        self.preview.toolbar.add_tool(remove_button)
        self.preview.toolbar.add_tool(save_button)

        self.wave_table_store = base_widgets.StoreComboBox(config.wave_tables_store, info_button=True)
        item = Gtk.ToolItem()
        item.add(self.wave_table_store)
        self.preview.toolbar.add(item)

    def initialize_notebook(self):
        for constituent in self.instrument.constituents:
            self.add_new_page(constituent)

    def add_constituent(self, source=None, constituent=None, weight=0, freq=1, shfit=0):
        if constituent is None:
            print(self.wave_table_store.get_active_value())
            constituent = copy.copy(self.wave_table_store.get_active_value())
        self.instrument.add_constituent(constituent, weight, freq, shfit)
        self.add_new_page(constituent)

    def add_new_page(self, constituent):
        try:
            self.notebook.append_page(_WaveTableEditorPage(self.instrument.wav_table_options, constituent),
                                      Gtk.Label(constituent.name))
        except:
            self.instrument.remove_constituent(constituent)
        self.show_all()

    def remove_constituent(self, source=None):
        constituent_ind = self.notebook.get_current_page()
        constituent = self.notebook.get_nth_page(constituent_ind).constituent
        self.notebook.remove_page(constituent_ind)

        self.instrument.remove_constituent(constituent)

    def update(self):
        self.instrument.update_wav_table()
        self.preview.refresh()

    def save(self, source=None):
        menu = base_widgets.StoreDialog(config.wave_tables_store, item=self.wave_table,
                                        action=Gtk.FileChooserAction.SAVE)
        response = menu.run()


class ModulationTable(Gtk.VBox):
    def __init__(self, instrument):
        Gtk.VBox.__init__(self)
        self.notebook = base_widgets.Notebook()
        self.notebook.set_show_border(False)

        self.instrument = instrument
        self.mod_tables = []

        self.freq_mod_amps = []
        self.amp_mod_amps = []
        self.freq_mod_freqs = []
        self.amp_mod_freqs = []

        self.add_button = Gtk.Button('add')
        self.remove_button = Gtk.Button('remove')

        add_button = imaging.NavigationToolButton('Add', 'Add other wave table', 'add', self.add_timbre)
        remove_button = imaging.NavigationToolButton('Remove', 'Remove constituent', 'remove', self.remove_timbre)
        save_button = imaging.NavigationToolButton('Save', 'Save Modulation Table', 'save', self.save)
        import_button = imaging.NavigationToolButton('Import', 'Import Modulation Table', 'import', self.import_timbre)

        self.action_toolbar = Gtk.Toolbar()
        self.action_toolbar.insert(add_button, -1)
        self.action_toolbar.insert(remove_button, -1)
        self.action_toolbar.insert(save_button, -1)
        self.action_toolbar.insert(import_button, -1)

        self.modulation_combo = base_widgets.StoreComboBox(config.modulation_store)
        item = Gtk.ToolItem()
        item.add(self.modulation_combo)
        self.action_toolbar.add(item)

        self.add(self.notebook)
        self.add(self.action_toolbar)
        self.import_from_timbre(self.instrument.timbre)

    def import_from_timbre(self, timbre):
        for tab in timbre.partial_waves:
            weight, ama, amf, harmonic, fma, fmf, shift = tab
            self.add_new_page(weight=weight, fma=fma, fmf=fmf, ama=ama, amf=amf, harmonic=harmonic, shift=shift)
        self.update()

    def add_new_page(self, weight=1, fma=0, fmf=0, ama=0, amf=0, harmonic=1, shift=0):
        mod_table = _ModulationTablePage(weight=weight,
                                         fma=fma,
                                         fmf=fmf,
                                         ama=ama,
                                         amf=amf,
                                         harmonic=harmonic,
                                         shift=shift)
        mod_table.props.halign = Gtk.Align.CENTER

        self.mod_tables.append(mod_table)
        self.notebook.append_page(mod_table)

        self.show_all()

    def add_timbre(self, source):
        self.add_new_page()

    def remove_timbre(self, source):
        if len(self.mod_tables) > 1:
            mod_table = self.notebook.remove_current_page()
            self.mod_tables.remove(mod_table)

    def update(self):
        self.instrument.update_osc_params(self.mod_tables)

    def save(self, source=None):
        menu = base_widgets.StoreDialog(config.modulation_store, item=self.instrument.timbre,
                                        action=Gtk.FileChooserAction.SAVE)
        response = menu.run()

    def import_timbre(self, source=None):
        active_timbre = self.modulation_combo.get_active_value()
        self.import_from_timbre(active_timbre)
        self.show_all()


class _ModulationTablePage(Gtk.Box):
    def __init__(self, weight=1, fma=0, fmf=0, ama=0, amf=0, harmonic=1, shift=0):
        Gtk.Box.__init__(self)

        self.options = {'weight': weight,
                        'fma': fma,
                        'ama': ama,
                        'fmf': fmf,
                        'amf': amf,
                        'harmonic': harmonic,
                        'shift': shift}

        modulation_table = Gtk.Grid()
        self.add(modulation_table)

        modulation_table.set_row_spacing(config.default_rowspacing)
        modulation_table.set_column_spacing(config.default_colspacing)

        self.fma = Gtk.Entry()
        self.fma.connect('changed', self.on_fma_entry_changed)
        self.fma.set_text(str(self.options['fma']))

        self.fmf = Gtk.Entry()
        self.fmf.connect('changed', self.on_fmf_entry_changed)
        self.fmf.set_text(str(self.options['fmf']))

        self.ama = Gtk.Entry()
        self.ama.connect('changed', self.on_ama_entry_changed)
        self.ama.set_text(str(self.options['ama']))

        self.amf = Gtk.Entry()
        self.amf.connect('changed', self.on_amf_entry_changed)
        self.amf.set_text(str(self.options['amf']))

        self.weight = Gtk.Entry()
        self.weight.connect('changed', self.on_weight_entry_changed)
        self.weight.set_text(str(self.options['weight']))

        self.harmonic = Gtk.Entry()
        self.harmonic.connect('changed', self.on_harmonic_entry_changed)
        self.harmonic.set_text(str(self.options['harmonic']))

        self.shift = Gtk.Entry()
        self.shift.connect('changed', self.on_shift_entry_changed)
        self.shift.set_text(str(self.options['shift']))

        modulation_table.add(Gtk.Label('FM amplitude'))
        modulation_table.attach(self.fma, 1, 0, 1, 1)
        modulation_table.attach(Gtk.Label('  FM frequency'), 2, 0, 1, 1)
        modulation_table.attach(self.fmf, 3, 0, 1, 1)
        modulation_table.attach(Gtk.Label('AM amplitude'), 0, 1, 1, 1)
        modulation_table.attach(self.ama, 1, 1, 1, 1)
        modulation_table.attach(Gtk.Label('  AM frequency'), 2, 1, 1, 1)
        modulation_table.attach(self.amf, 3, 1, 1, 1)
        modulation_table.attach(Gtk.Label('Weight'), 0, 2, 1, 1)
        modulation_table.attach(self.weight, 1, 2, 1, 1)
        modulation_table.attach(Gtk.Label('Harmonic'), 2, 2, 1, 1)
        modulation_table.attach(self.harmonic, 3, 2, 1, 1)
        modulation_table.attach(Gtk.Label('Shift'), 2, 3, 1, 1)
        modulation_table.attach(self.shift, 3, 3, 1, 1)

    def on_fma_entry_changed(self, source):
        try:
            self.options['fma'] = gtk_utils.read_entry(self.fma, float)
        except:
            self.options['fma'] = 0
            self.fma.set_text('0')

    def on_fmf_entry_changed(self, source):
        try:
            self.options['fmf'] = gtk_utils.read_entry(self.fmf, float)
        except:
            self.options['fmf'] = 0
            self.fmf.set_text('0')

    def on_ama_entry_changed(self, source):
        try:
            self.options['ama'] = gtk_utils.read_entry(self.ama, float)
        except:
            self.options['ama'] = 0
            self.ama.set_text('0')

    def on_amf_entry_changed(self, source):
        try:
            self.options['amf'] = gtk_utils.read_entry(self.amf, float)
        except:
            self.options['amf'] = 0
            self.amf.set_text('0')

    def on_harmonic_entry_changed(self, source):
        try:
            self.options['harmonic'] = gtk_utils.read_entry(self.harmonic, float)
        except:
            self.options['harmonic'] = 1
            self.harmonic.set_text('1')

    def on_weight_entry_changed(self, source):
        try:
            self.options['weight'] = gtk_utils.read_entry(self.weight, float)
        except:
            self.options['weight'] = 1
            self.weight.set_text('1')

    def on_shift_entry_changed(self, source):
        try:
            self.options['shift'] = gtk_utils.read_entry(self.shift, float)
        except:
            self.options['shift'] = 0
            self.shift.set_text('0')

class _WaveTableEditorPage(Gtk.VBox):
    def __init__(self, options, constituent):
        Gtk.VBox.__init__(self)
        self.options = options
        self.constituent = constituent

        content_area = Gtk.VBox()
        content_area.set_border_width(5)
        self.pack_start(content_area, True, True, 0)

        self.weight_scale = Gtk.HScrollbar()
        self.weight_scale.set_range(0, 1 * 1000)
        self.weight_scale.set_increments(0.01, 0.01)
        self.weight_scale.set_value(self.options['weights'][self.constituent] * 1000)
        self.weight_scale.connect('value_changed', self.on_weight_scale_value_changed)
        content_area.pack_start(self.weight_scale, True, True, 5)

        self.freq_scale = Gtk.HScrollbar()
        self.freq_scale.set_range(1, 10)
        self.freq_scale.set_increments(1, 1)
        self.freq_scale.set_value(self.options['freqs'][self.constituent])
        self.freq_scale.connect('value_changed', self.on_freq_scale_value_changed)
        content_area.pack_start(self.freq_scale, True, True, 5)

        self.shift_scale = Gtk.HScrollbar()
        self.shift_scale.set_range(0, 10 * 1000)
        self.shift_scale.set_increments(0.01, 0.01)
        self.shift_scale.set_value(self.options['shifts'][self.constituent] * 1000)
        self.shift_scale.connect('value_changed', self.on_shift_scale_value_changed)
        content_area.pack_start(self.shift_scale, True, True, 5)

    ######### UI signals ###############################################################################################

    def on_weight_scale_value_changed(self, source):
        self.options['weights'][self.constituent] = self.weight_scale.get_value() / 1000

    def on_shift_scale_value_changed(self, source):
        self.options['shifts'][self.constituent] = self.shift_scale.get_value() / 1000

    def on_freq_scale_value_changed(self, source):
        value = int(self.freq_scale.get_value())
        source.set_value(value)
        self.options['freqs'][self.constituent] = value


class _FrequencyMapTransitionEditor(GraphEditor):
    def __init__(self, freq_map_transition, *args, **kwargs):
        GraphEditor.__init__(self, *args, **kwargs)

        self.freq_map_transition = freq_map_transition

        self.set_new_graph(self.freq_map_transition, self.draw_transition_map, self.update_transition_map)
        self.set_insert_point_rules(self.insert_point_rules)
        self.set_mouse_sensibility(0.1, 0.02)

        self.time_pointer = imaging.TimePointer(self.graph_plot, config.player)
        self.refresh()

    def insert_point_rules(self, coord_data, *args, **kwargs):
        y, x = coord_data
        return y, min(max(x, 0), 1)

    def draw_transition_map(self, ax, data, **kwargs):
        coords, coords_smoothed = data
        ys, xs = coords
        ys_smoothed, xs_smoothed = coords_smoothed

        collection = ax.scatter(xs, ys)
        line, = ax.plot(xs_smoothed, ys_smoothed)
        ax.set_ylim([-.1, 1.1])
        return collection, line

    def update_transition_map(self, ax, artists, data):
        coords, coords_smoothed = data
        ys, xs = coords
        offsets = np.array([xs, ys]).T
        ys_smoothed, xs_smoothed = coords_smoothed
        collection, line = artists

        collection.set_offsets(offsets)
        line.set_data(xs_smoothed, ys_smoothed)
        return collection, line


class _FrequencyMapEditor(GraphEditor):
    def __init__(self, freq_map, *args, **kwargs):
        GraphEditor.__init__(self, *args, **kwargs)

        self.freq_map = freq_map

        self.graph_plot.toolbar.remove_tool_by_name('Insert/Remove')

        self.set_new_graph(self.freq_map, self.draw_map, self.update_map)
        self.set_insert_point_rules(self.insert_point_rules)
        self.set_mouse_sensibility(100, 10)
        self.refresh()

    def insert_point_rules(self, coord_data, orig_data_point, *args, **kwargs):
        y, x = coord_data
        y_orig, x_orig = orig_data_point
        return max(0, y), x_orig

    def draw_map(self, ax, data, **kwargs):
        coords, coords_smoothed = data
        ys, xs = coords
        line, = ax.step(xs, ys)
        return line

    def update_map(self, ax, artists, data):
        coords, coords_smoothed = data
        ys, xs = coords
        line = artists
        line.set_ydata(ys)
        return line


class FrequencyMapArrayEditor(TrackEditor):
    __gsignals__ = {"reset_track": (GObject.SignalFlags.RUN_LAST, None, [GObject.TYPE_PYOBJECT])}

    def __init__(self, track, min_dims=None):
        TrackEditor.__init__(self)
        scrolled = base_widgets.CollapsingSectionWindow(min_dims=min_dims)
        self.add(scrolled)

        self.nr_subeditors = 0

        self.freq_map_array = track.freq_map_array
        self.fill_with_editors(scrolled)

        self.show_all()

    def fill_with_editors(self, scrolled):
        for i, tuple in enumerate(self.freq_map_array.map_tuples):
            start_map_editor = _FrequencyMapEditor(tuple.start_map, dims=config.dims_freq_map_editor)
            scrolled.add_child(start_map_editor, border=config.section_border, hborder=config.default_hborder,
                               rubberband=config.section_rubberband, title_label=Gtk.Label('Initial Frequency Map'))
            self.nr_subeditors += 1
            if not tuple.is_unique:
                transition_editor = _FrequencyMapTransitionEditor(tuple.transition, dims=config.dims_freq_map_editor)
                scrolled.add_child(transition_editor, border=config.section_border, hborder=config.default_hborder,
                                   rubberband=config.section_rubberband, title_label=Gtk.Label('Transition'))
                self.nr_subeditors += 1

                if i + 1 == len(self.freq_map_array.map_tuples):
                    end_map_editor = _FrequencyMapEditor(tuple.end_map, dims=config.dims_freq_map_editor)
                    scrolled.add_child(end_map_editor, border=config.section_border, hborder=config.default_hborder,
                                       rubberband=config.section_rubberband,
                                       title_label=Gtk.Label('Final Frequency Map'))
                    self.nr_subeditors += 1

            scrolled.show_all()

    def update(self):
        self.emit('reset_track', self.track)


class BufferView(imaging.UpdatedGraphPlot):
    def __init__(self, buff, *args, **kwargs):
        imaging.UpdatedGraphPlot.__init__(self, *args, **kwargs)

        if config.params.channels == 1:
            self.buffer_graph = graph.Graph(buff.arr, np.linspace(0, 1, len(buff.arr)),
                                            'buffer_view', raw_data=True, settable=False)

            length = len(self.buffer_graph.get_raw_xs())
            self.set_x_mirr(1)
            self.set_new(self.buffer_graph, self.draw_buffer, self.update_buffer)

        elif config.params.channels == 2:
            self.buffer_graph_left = graph.Graph(buff.arr_left, np.linspace(0, 1, len(buff.arr_left)),
                                            'buffer_view_left', raw_data=True, settable=False)

            self.buffer_graph_right = graph.Graph(buff.arr_right, np.linspace(0, 1, len(buff.arr_right)),
                                                 'buffer_view_right', raw_data=True, settable=False)

            length = len(self.buffer_graph_right.get_raw_xs())
            self.set_x_mirr(1)
            self.set_new(self.buffer_graph_left, self.draw_buffer_left, self.update_buffer)
            self.set_new(self.buffer_graph_right, self.draw_buffer_right, self.update_buffer)

        self.refresh()

        imaging.TimePointer(self, config.player, 1, *args, **kwargs)
        config.player.sound_gen_thread.connect('idle', self.refresh_buffer)

    def draw_buffer_left(self, ax, data, **kwargs):
        return self.draw_buffer(ax, data, color='blue')

    def draw_buffer_right(self, ax, data, **kwargs):
        return self.draw_buffer(ax, data, color='red')

    def draw_buffer(self, ax, data, color='blue', **kwargs):
        coords_canvas, coords_canvas_smoothed = data
        ys, xs = coords_canvas

        line, = ax.plot(xs, ys, color=color, alpha=0.4)
        # ax.axis('off')
        if self.orientation == 1 :
            ax.set_ylim([0, max(ys)])
            ax.set_xlim([-5, 5])
        else:
            ax.set_xlim([0, max(xs)])
            ax.set_ylim([-5, 5])
        return line

    def update_buffer(self, ax, line, data):
        coords_canvas, coords_canvas_smoothed = data
        ys, xs = coords_canvas
        line.set_data(xs, ys)
        # if self.orientation == 1:
        #     ax.set_xlim([min(xs), max(xs)])
        # else:
        #     ax.set_ylim([min(ys), max(ys)])
        # ax.relim()
        # ax.autoscale_view()
        return line

    def refresh_buffer(self, source=None):
        if config.params.channels == 1:
            self.refresh_data_obj(self.buffer_graph)
        else:
            self.refresh_data_obj(self.buffer_graph_left)
            self.refresh_data_obj(self.buffer_graph_right)


class ColorSyntaxEditor(Gtk.Box):
    pass