import data_structures.image as image
import run_time.config as config
import run_time.py_utils as py_utils

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GLib, GObject

import os
import numpy as np

import matplotlib
matplotlib.use('GTK3Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
# from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo as FigureCanvas
from matplotlib.backends.backend_gtk3 import NavigationToolbar2GTK3
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt


class NavigationToolbar(NavigationToolbar2GTK3):
    def __init__(self, updated_figure, window):
        """
        Wrapper of matplotlibs NavigationToolbar for Gtk. Removes unnessecary tools, allows easy adding and removal
        of new tools and fixes a bug in matplotlibs rubberband drawing (see draw_rubberband).

        :param updated_figure:
        :param window:
        """
        NavigationToolbar2GTK3.__init__(self, updated_figure.canvas, window)

        # make toolitems extensible
        self.toolitems = list(self.toolitems)

        # remove unnecessary tools
        self.remove_tool_by_name('Save')
        self.remove_tool_by_name('Subplots')

        self.ax = updated_figure.ax
        self.updated_figure = updated_figure

        # implement the zooming rubberband as an overlay
        self.zoom_rubberband_overlay = Rubberband(updated_figure)
        self.zoom_rubberband_bbox = None

        self._hide_message = False

    def set_cursor(self, cursor):
        return None

    def add_tool(self, tool_button, ind=-1):
        self.insert(tool_button, ind)
        self.toolitems.insert(ind, tool_button.description)

    def add_tool_on_name(self, tool_button, name):
        ind = self.get_toolitem_ind(name)
        self.add_tool(tool_button, ind)

    def remove_tool_by_name(self, name):
        ind = self.get_toolitem_ind(name)
        self.remove(self.get_children()[ind])

    def draw_rubberband(self, event, x0, y0, x1, y1):
        """
        This is a fix of the library method as it uses cairo which changes the backend.

        :param event:
        :param x0:
        :param y0:
        :param x1:
        :param y1:
        :return:
        """
        self.zoom_rubberband_bbox = x0, y0, x1, y1
        self.zoom_rubberband_overlay.refresh(rubberband_bbox=self.zoom_rubberband_bbox)

    def get_toolitem_ind(self, name):
        return [description[0] for description in self.toolitems].index(name)

    def remove_rubberband(self):
        self.zoom_rubberband_overlay.hide()
        self.zoom_rubberband_bbox = None

    def hide_message(self, bool):
        self._hide_message = bool

    def set_message(self, s):
        if not self._hide_message:
            super(NavigationToolbar, self).set_message(s)
        else:
            pass


class NavigationToolbarAction(object):
    def __init__(self, toolbar, _action_id, _action_message, press_connect, release_connect):
        """
        Defines an action that can be wrapped by a NavigationToggleToolButton. An action can be defined by the two
        callbacks press_connect and release_connect.

        :param toolbar:
        :param _action_id: str
        :param _action_message: str
        :param press_connect: callable
        :param release_connect: callable
        """
        self._action_id = _action_id
        self._action_message = _action_message
        self.press_connect = press_connect
        self.release_connect = release_connect

        self.toolbar = toolbar

    def set_activated(self, source=None):
        if self.toolbar._active == self._action_id:
            self.toolbar._active = None
        else:
            self.toolbar._active = self._action_id

        if self.toolbar._idPress is not None:
            self.toolbar._idPress = self.toolbar.canvas.mpl_disconnect(self.toolbar._idPress)
            self.toolbar.mode = ''

        if self.toolbar._idRelease is not None:
            self.toolbar._idRelease = self.toolbar.canvas.mpl_disconnect(self.toolbar._idRelease)
            self.toolbar.mode = ''

        if self.toolbar._active:
            if self.press_connect is not None:
                self.toolbar._idPress = self.toolbar.canvas.mpl_connect('button_press_event', self.press_connect)
            else:
                self.toolbar._idPress = None

            if self.release_connect is not None:
                self.toolbar._idRelease = self.toolbar.canvas.mpl_connect('button_release_event', self.release_connect)
            else:
                self.toolbar._idRelease = None

            self.toolbar.mode = self._action_message
            try:
                self.toolbar.canvas.widgetlock(self.toolbar)
            except ValueError:
                pass
        else:
            self.toolbar.canvas.widgetlock.release(self.toolbar)

        for a in self.toolbar.canvas.figure.get_axes():
            a.set_navigate_mode(self.toolbar._active)

        self.toolbar.set_message(self.toolbar.mode)


class NavigationToolButton(Gtk.ToolButton):
    def __init__(self, text, tooltip_text, image_file, callback):
        """
        A button in a NavigationToolbar that calls callback when clicked. To be added to a NavigationToolbar.
        :param text:
        :param tooltip_text:
        :param image_file:
        :param callback:
        """
        Gtk.ToolButton.__init__(self)

        self.description = text, tooltip_text, image_file, callback

        basedir = os.path.join(config.icons_path, 'navigator_toolbar')

        try:
            fname = os.path.join(basedir, image_file + '.png')
            image = Gtk.Image()
            image.set_from_file(fname)
        except:
            pass

        self.set_label(text)
        self.set_icon_widget(image)
        self.connect('clicked', callback)
        self.set_tooltip_text(tooltip_text)


class NavigationToggleToolButton(Gtk.ToolButton):
    def __init__(self, text, tooltip_text, image_file1, image_file2, action1, action2):
        """
        A button in a NavigationToolbar that activates an action. The action object decides when the action is performed.

        :param text:
        :param tooltip_text:
        :param image_file1:
        :param image_file2:
        :param action1: NavigationToolbarAction
        :param action2: NavigationToolbarAction
        """
        Gtk.ToolButton.__init__(self)

        self.description = text, tooltip_text, image_file1, image_file2, action1, action2

        self.action1 = action1
        self.action2 = action2
        self._active_action = action1

        basedir = os.path.join(config.icons_path, 'navigator_toolbar')

        fname = os.path.join(basedir, image_file1 + '.png')
        self.image1 = Gtk.Image()
        self.image1.set_from_file(fname)

        fname = os.path.join(basedir, image_file2 + '.png')
        self.image2 = Gtk.Image()
        self.image2.set_from_file(fname)

        self.set_label(text)
        self.connect('clicked', self.toggle_actions)
        self.set_tooltip_text(tooltip_text)

        self.set_action1_active()

    def toggle_actions(self, source=None):
        if self._active_action == self.action2:
            self.set_action1_active()
        else:
            self.set_action2_active()

    def set_action1_active(self):
        self.set_icon_widget(self.image1)
        self.action1.set_activated()
        self._active_action = self.action1
        self.show_all()

    def set_action2_active(self):
        self.set_icon_widget(self.image2)
        self.action2.set_activated()
        self._active_action = self.action2
        self.show_all()


class GraphNavigationToolbar(NavigationToolbar):
    def __init__(self, graph_editor, active_graph, window):
        """
        A NavigationToolbar object for GraphEditor objects.

        :param graph_editor:
        :param active_graph:
        :param window:
        """
        NavigationToolbar.__init__(self, graph_editor.graph_plot, window)
        self.editor = graph_editor
        self.graph_plot = graph_editor.graph_plot
        self.graph = active_graph

        action_insert = NavigationToolbarAction(self, 'INSERT', 'insert', self.press_insert, None)
        action_remove = NavigationToolbarAction(self, 'REMOVE', 'remove', self.press_remove, None)
        insert_remove_button = NavigationToggleToolButton('Insert/Remove', 'Insert or remove data points',
                                                          'add', 'remove', action_insert, action_remove)
        self.add_tool_on_name(insert_remove_button, 'Zoom')

        action_move = NavigationToolbarAction(self, 'MOVE', 'move', self.press_move, self.release_move)
        move_button = NavigationToolButton('Move', 'Move data points', 'move', action_move.set_activated)
        self.add_tool_on_name(move_button, 'Insert/Remove')

        self._moving_data_point = None
        self._moving_data_point_ind = None
        self._ids_move = None

        self.show_all()

    def _get_closest_data_point(self, coord_data):
        """
        Finds the closest data point of graph to coord_data

        :param coord_data:
        :return: (index, distance), where index is the ith data point in graph
        """
        dist_2 = (self.graph.get_raw_ys() - coord_data[0])**2 + (self.graph.get_raw_xs() - coord_data[1])**2
        i = np.argmin(dist_2)
        min_dist = np.sqrt(dist_2[i])
        return i, min_dist

    def _is_coord_near_data_point(self, coord, ind):
        if coord[1] - self.graph.get_raw_xs()[ind] < self.editor.press_data_point_threshold_x and \
                coord[0] - self.graph.get_raw_ys()[ind] < self.editor.press_data_point_threshold_y:
            return True
        return False

    def _can_be_changed(self, graph):
        """
        Raw graphs can only be changed by moving data points. Insertion and removal of data points is only allowed
        when self.current_action == 'move_data_point' in this case.
        :param graph:
        :return:
        """
        return not graph.is_raw_data or (graph.is_raw_data and self._active == 'MOVE')

    def _insert_point(self, event):
        if self._can_be_changed(self.graph):
            coord = self.graph_plot.map_coord_canvas_to_data(event)
            y, x = self.editor.apply_insert_point_rules(coord, self._moving_data_point)
            i = self.graph.insert(y, x)
            return i
        return None

    def _remove_point(self, ind):
        if self._can_be_changed(self.graph):
            self.graph.remove(ind)

    def press_remove(self, event):
        coord_data = self.graph_plot.map_coord_canvas_to_data(event)
        ind, dist = self._get_closest_data_point( coord_data)
        if self._is_coord_near_data_point(coord_data, ind):
            self._remove_point(ind)
            self.draw()

        self.graph_plot.refresh_data_obj(self.graph)
        self.press(event)

    def press_move(self, event):
        coord_data = self.graph_plot.map_coord_canvas_to_data(event)
        ind, dist = self._get_closest_data_point(coord_data)
        if self._is_coord_near_data_point(coord_data, ind):
            self._moving_data_point = (self.graph.get_raw_ys()[ind], self.graph.get_raw_xs()[ind])
            self._moving_data_point_ind = ind

            id1 = self.graph_plot.canvas.mpl_connect('motion_notify_event', self.drag_move)
            id2 = self.graph_plot.canvas.mpl_connect('figure_leave_event', self.leave_move)
            self._ids_move = id1, id2

        self.press(event)

    def drag_move(self, event):
        if self._moving_data_point_ind is not None:
            self._remove_point(self._moving_data_point_ind)

            # in case event is forbidden by graph.apply_insert_point_rules, put back the data point
            try:
                ind = self._insert_point(event)
                self._moving_data_point_ind = ind
            except AssertionError:
                self._insert_point(self._moving_data_point)
                self.release_move()

            self.graph_plot.refresh_data_obj(self.graph)

    def release_move(self, event=None):
        self._ids_move = None
        self._moving_data_point = None
        self._moving_data_point_ind = None

    def leave_move(self, event=None):
        if self._moving_data_point is not None:
            self._insert_point(self._moving_data_point)
            self.release_move()

    def press_insert(self, event):
        try:
            self._insert_point(event)
        except:
            pass

        self.graph_plot.refresh_data_obj(self.graph)
        self.press(event)


class Updater(object):
    def __init__(self):
        self._drawn_artists = None
        self._draw_precedence = None
        self.refresh_timer = None

        self.canvas = None

    def has_draw_precedence(self, obj):
        return obj == self._draw_precedence

    def set_draw_precedence(self, obj, val):
        if self.has_draw_precedence(obj) or not self.is_draw_fixed():
            self._draw_precedence = val

    def is_draw_fixed(self):
        return self._draw_precedence is not None

    def drawer(self, ax, *args, **kwargs):
        """
        Function that draws on ax initially and returns an object pointing to the drawn artists
        that is updated with self.updater
        :param ax:
        :return: object
        """
        raise NotImplementedError

    def updater(self, ax, artists, *args, **kwargs):
        """
        Function that updates the artists first drawn with drawer
        :param ax:
        :param artists:
        :return:
        """
        raise NotImplementedError

    def redraw(self):
        """
        Force redrawing of all artists.
        :return:
        """
        self._drawn_artists = None
        self.refresh()

    def refresh(self, source=None, artists=None, *args, **kwargs):
        """
        Renders all the artists. Returns True on success and False upon exception. This is needed to start GTK managed
        threads by start since GObject.idle_add is called as long as True is returned.
        :param artists:
        :param source:
        :param args:
        :param kwargs:
        :return:
        """
        if artists is None:
            artists = self._drawn_artists

        try:
            self.render_artists(artists, *args, **kwargs)

            if (self.is_draw_fixed() and self.has_draw_precedence(self)) or not self.is_draw_fixed():
                self.canvas.draw_idle()
                # print 'self', self
                # print 'fixed', self.is_draw_fixed()
                # try:
                #     print self.updated_figure._draw_precedence
                # except:
                #     pass
                # traceback.print_stack()
                # print

        except Exception as e:
            print(e)

    def render_artists(self, artists, *args, **kwargs):
        """
        Either update artists or draw them for the first time.
        :param artists:
        :param args:
        :param kwargs:
        :return:
        """
        must_be_drawn = artists is None
        if must_be_drawn:
            self.first_draw(*args, **kwargs)
        else:
            self.update(artists, *args, **kwargs)

    def first_draw(self, *args, **kwargs):
        self._drawn_artists = self.drawer(self.ax, *args, **kwargs)

    def update(self, artists, *args, **kwargs):
        self.updater(self.ax, artists, *args, **kwargs)

    def set_drawer(self, draw_callable, update_callable):
        self.drawer = draw_callable
        self.updater = update_callable

    def remove_artists(self, artists):
        for artist in py_utils.flatten(artists):
            artist.remove()
            del artist

    def set_artists_visible(self, bool, artists):
        for artist in py_utils.flatten(artists):
            artist.set_visible(bool)

    def start_refreshing_loop(self, source=None):
        self.refresh_timer = self.canvas.new_timer(interval=120)
        self.refresh_timer.add_callback(self.refresh)
        self.refresh_timer.start()

        # self.set_draw_precedence(self, self)

        # self.refresh_timer = thread.PeriodicFunctionCall(0.15, self.refresh, True)
        # self.refresh_timer.start()

    def stop_refreshing_loop(self, source=None):
        try:
            self.refresh_timer.stop()
            # self.set_draw_precedence(self, None)
        except AttributeError as e:
            pass
        self.refresh()


class UpdatedFigure(Gtk.VBox, Updater):
    def __init__(self, dims, dpi=30):
        """
        Any type of graphical display should derive from this class. It implements matplotlib figure that can be
        efficiently updated
        :param dims:
        :param dpi:
        """
        Gtk.VBox.__init__(self)
        Updater.__init__(self)

        self.dims = dims
        self.dpi = dpi

        self.figure, self.ax, self.canvas = self.initialize_plt_objects(dims)

        self.content_area = Gtk.Box()
        self.toolbar = NavigationToolbar(self, config.main_window)
        self.content_area.pack_start(self.toolbar, True, True, 0)

        self.add(self.canvas)
        self.pack_start(self.content_area, False, False, 0)

    def remove_toolbar(self):
        self.content_area.remove(self.toolbar)

    def initialize_plt_objects(self, dims):
        dims_in_inches = self.get_dims_in_inches(dims)
        figure = Figure(figsize=dims_in_inches, dpi=self.dpi)

        ax = plt.Axes(figure, [0., 0., 1., 1.])
        ax.set_axis_off()
        figure.add_axes(ax)

        canvas = FigureCanvas(figure)
        canvas.set_size_request(dims[1], dims[0])

        return figure, ax, canvas

    def reset(self):
        self.figure, self.ax, self.canvas = self.initialize_plt_objects(self.dims)
        self._drawn_artists = None

    def get_preview(self, dims):
        figure, ax, canvas = self.resize(dims, in_place=False)
        self.refresh()
        return canvas

    def get_dims_in_inches(self, dims):
        return dims[0] / float(self.dpi), dims[1] / float(self.dpi)

    def resize(self, dims, in_place=True):
        """
        Resize the canvas to dims. If in_place=True, this instance is resized.
        :param dims:
        :param in_place:
        :return:
        """
        if in_place:
            if dims != self.dims:
                self.dims = dims
                dims_in_inches = self.get_dims_in_inches(dims)
                self.figure.set_size_inches(dims_in_inches, forward=True)
                self.canvas.set_size_request(int(self.dims[1]), int(self.dims[0]))
        else:
            return self.initialize_plt_objects(dims)


class DataFigure(UpdatedFigure):
    def __init__(self, *args, **kwargs):
        UpdatedFigure.__init__(self, *args, **kwargs)

        self.data_objects = []
        self.drawers = {}
        self.updaters = {}
        self._drawn_artists = {}

    def drawer(self, ax, data_obj, *args, **kwargs):
        self._drawn_artists[data_obj] = self.drawers[data_obj](ax, self.get(data_obj), *args, **kwargs)
        return self._drawn_artists

    def updater(self, ax, artists, data_obj, *args, **kwargs):
        self.updaters[data_obj](ax, artists, self.get(data_obj), *args, **kwargs)

    def get(self, data_object):
        """
        Function that extracts the data relevant to to drawers and updaters from data object. Specifically, its return
        value is passed to the drawer and updater functions.

            self.drawers[data_obj](ax, self.get(data_obj), *args, **kwargs)
            self.updaters[data_obj](ax, artists[data_obj], self.get(data_obj), *args, **kwargs)

        This allows the implementation of derived classes with customized data transformations as these transformations
        can be implemented on plotting level and must not be defined in the classes holding the data.

        :param data_object:
        :return:
        """
        return data_object

    def set_new(self, data_obj, drawer, updater):
        self.data_objects.append(data_obj)
        self.drawers[data_obj] = drawer
        self.updaters[data_obj] = updater
        self._drawn_artists[data_obj] = None
        self.refresh_data_obj(data_obj)

    def redraw(self):
        for k, v in self._drawn_artists.items():
            self._drawn_artists[k] = None
            for vv in py_utils.flatten(v):
                vv.remove()
        self.refresh()

    def refresh(self, source=None, *args, **kwargs):
        for data_obj in self.data_objects:
            self.refresh_data_obj(data_obj, draw=False, *args, **kwargs)
        self.canvas.draw_idle()

    def refresh_data_obj(self, data_obj, draw=True, *args, **kwargs):
        self.render_artists(self._drawn_artists[data_obj], data_obj, *args, **kwargs)
        if draw:
            self.canvas.draw_idle()


class UpdatedGraphPlot(DataFigure):
    def __init__(self, orientation=0, direction_x=1, direction_y=1, *args, **kwargs):
        DataFigure.__init__(self, *args, **kwargs)

        self.active_graph = None
        self.direction_x = direction_x
        self.direction_y = direction_y
        self.orientation = orientation

        self.x_mirr_coord = None

    def get(self, graph):
        coords_canvas_smoothed = None, None
        if not graph.is_raw_data:
            coords_canvas_smoothed = self._map_coord_data_to_canvas(*graph.get())

        coords_canvas = self._map_coord_data_to_canvas(*graph.get_raw_graph())
        return coords_canvas, coords_canvas_smoothed

    @property
    def graphs(self):
        return self.data_objects

    def refresh_active_graph(self):
        self.refresh_data_obj(self.active_graph)

    def set_x_mirr(self, x_mirr_coord):
        self.x_mirr_coord = x_mirr_coord

    def _map_coord_data_to_canvas(self, y_data, x_data):
        # y_data = np.array(y_data)
        # x_data = np.array(x_data)

        y_mirr, x_mirr = self._mirror_y_and_x(y_data, x_data)
        y, x = self._invert_y_and_x(y_mirr, x_mirr)

        return y, x

    def _map_coord_canvas_to_data(self, y_canvas, x_canvas):
        y_canvas = np.array(y_canvas)
        x_canvas = np.array(x_canvas)

        y, x = self._invert_y_and_x(y_canvas, x_canvas)
        y_mirr, x_mirr = self._mirror_y_and_x(y, x)

        return y_mirr, x_mirr

    def _invert_y_and_x(self, y, x):
        if self.orientation == 0:
            return y, x
        elif self.orientation == 1:
            return x, y
        else:
            raise ValueError('kwarg orientation must be 0 or 1')

    def _mirror_y_and_x(self, y, x):
        y_mirr = self.direction_y * y
        if self.direction_x == -1:
            x_mirr = self.x_mirr_coord - x
        else:
            x_mirr = x

        return y_mirr, x_mirr

    def map_coord_canvas_to_data(self, event):
        if isinstance(event, tuple):
            ydata, xdata = event
        else:
            ydata, xdata = event.ydata, event.xdata

        return self._map_coord_canvas_to_data(ydata, xdata)


class UpdatedImage(UpdatedFigure):
    def __init__(self, imag=None, dims_canvas=None, dpi=100, *args, **kwargs):
        if imag is not None:
            self.imag = imag
        else:
            self.imag = image.Image(*args, **kwargs)

        if dims_canvas is None:
            dims_canvas = self.imag.shape

        UpdatedFigure.__init__(self, dims=dims_canvas, dpi=dpi)

        self.is_changed = False
        self.refresh()
        self.show_all()

    def reset(self, *args, **kwargs):
        self.imag.reset(*args, **kwargs)
        self.refresh()
        self.is_changed = False

    def extend(self, new_shape):
        return self.imag.extend(new_shape)

    def drawer(self, ax):
        ax_image = ax.imshow(self.imag.img,
                             extent=[0, self.imag.img.shape[1], 0, self.imag.img.shape[0]],
                             aspect='auto',
                             origin='lower')
        return ax_image

    def updater(self, ax, artist):
        artist.set_data(self.imag.img)
        # ax.set_xlim([-0.5, self.imag.img.shape[1]-0.5])
        # ax.set_ylim([-0.5, self.imag.img.shape[0]-0.5])
        return artist


class Canvas(UpdatedImage):
    def __init__(self, *args, **kwargs):
        """
        An interactive drawing area. All signals are passed to the underlying imag.

        :param imag: either instance of Image or TrackView, must have to_pixbuf, apply_brush and shape properties
        :param img_path: path to image readable by cv2.imread
        """
        # Load image or create empty
        UpdatedImage.__init__(self, *args, **kwargs)

        self.connect('button_press_event', self.on_canvas_button_press)
        self._current_brush = None
        self.is_changed = False
        self.current_selection = self.imag.img != None

        self.brush_preview = BrushPreview(self)

    @property
    def current_brush(self):
        return self._current_brush

    @current_brush.setter
    def current_brush(self, val):
        self._current_brush = val

    @staticmethod
    def map_imag_canvas(event):
        return int(event.ydata), int(event.xdata)

    def connect(self, *args, **kwargs):
        self.figure.canvas.mpl_connect(*args, **kwargs)

    def is_in_draw_mode(self):
        return self.toolbar._active is None and self.current_brush is not None

    ######### UI signals ###############################################################################################
    def on_canvas_button_press(self, event):
        if self.is_in_draw_mode() and self.current_brush is not None:
            event = self.map_imag_canvas(event)
            self.current_brush.apply(event, self.imag)
            self.refresh()
            self.is_changed = True


class Overlay(Updater):
    def __init__(self, updated_figure, drawer=None, updater=None, *args, **kwargs):
        Updater.__init__(self)

        self.updated_figure = updated_figure
        self.figure, self.canvas, self.ax = updated_figure.figure, updated_figure.canvas, updated_figure.ax

        if drawer is not None and updater is not None:
            self.set_drawer(drawer, updater)

        self.is_hidden = True

    def has_draw_precedence(self, obj):
        return self.updated_figure.has_draw_precedence(obj)

    def set_draw_precedence(self, obj, val):
        self.updated_figure.set_draw_precedence(obj, val)

    def is_draw_fixed(self):
        return self.updated_figure.is_draw_fixed()

    def refresh(self, *args, **kwargs):
        if self.is_hidden:
            self.show()
        if self.updated_figure.is_visible():
            super(Overlay, self).refresh(*args, **kwargs)

    def hide(self):
        if self._drawn_artists is not None:
            self.set_artists_visible(False, self._drawn_artists)
        self.is_hidden = True

    def show(self):
        if self._drawn_artists is not None:
            self.set_artists_visible(True, self._drawn_artists)
        self.is_hidden = False

    def reset(self):
        self.remove_artists(self._drawn_artists)


class Rubberband(Overlay):
    def get_rubberband_bounds(self, ax, rubberband_bbox):
        x0, y0, x1, y1 = rubberband_bbox

        axis_to_data = ax.transData.inverted()
        x0, y0 = axis_to_data.transform((x0, y0))
        x1, y1 = axis_to_data.transform((x1, y1))

        w = abs(x1 - x0)
        h = abs(y1 - y0)
        x = min(x0, x1)
        y = min(y0, y1)

        return x, y, w, h

    def drawer(self, ax, rubberband_bbox):
        x, y, w, h = self.get_rubberband_bounds(ax, rubberband_bbox)
        rect = Rectangle((x, y), width=w, height=h, fill=False, ec='red', lw=0.5)
        rect.set_zorder(100)
        return rect

    def updater(self, ax, artists, rubberband_bbox):
        rect = artists
        rect.set_bounds(*self.get_rubberband_bounds(ax, rubberband_bbox))
        ax.add_artist(rect)
        return rect


class TimePointer(Overlay):
    def __init__(self, updated_figure, player, total_time_in_figure_coords=1, orientation=0, direction_x=1, *args, **kwargs):
        Overlay.__init__(self, updated_figure, *args, **kwargs)
        self.total_time_in_figure_coords = total_time_in_figure_coords

        self.orientation = orientation
        self.direction = direction_x

        self.player = player

        self.player.connect('start_playing', self.start_refreshing_loop)
        self.player.connect('stop_playing', self.stop_refreshing_loop)

        self.refresh()

    def drawer(self, ax):
        where = self.get_time_offset()
        if self.orientation == 0:
            line = ax.axvline(where)
        else:
            line = ax.axhline(where)

        return line

    def updater(self, ax, artists):
        line = artists
        where = self.get_time_offset()
        if self.orientation == 0:
            line.set_xdata(where)
        else:
            line.set_ydata(where)

        return line

    def get_time_offset(self):
        if self.direction == 1:
            percent = self.player.time_in_percent
        else:
            percent = 1 - self.player.time_in_percent

        return percent * self.total_time_in_figure_coords


class BrushPreview(Overlay):
    def __init__(self, *args, **kwargs):
        Overlay.__init__(self, *args, **kwargs)
        self.updated_canvas.connect('motion_notify_event', self.update_preview)
        self.updated_canvas.connect('figure_enter_event', self.on_figure_enter)
        self.updated_canvas.connect('figure_leave_event', self.on_figure_leave)
        self.last_bounds = None

    @property
    def current_brush(self):
        return self.updated_canvas.current_brush

    @property
    def updated_canvas(self):
        return self.updated_figure

    def drawer(self, ax, bounds):
        xmin, xmax, ymin, ymax = bounds

        ymin_line = ax.axhline(ymin, color=config.brush_preview_color, lw=0.5)
        ymax_line = ax.axhline(ymax, color=config.brush_preview_color, lw=0.5)
        xmin_line = ax.axvline(xmin, color=config.brush_preview_color, lw=0.5)
        xmax_line = ax.axvline(xmax, color=config.brush_preview_color, lw=0.5)

        # rect = Rectangle((xmin, ymin), width=w, height=h, fill=False, ec='red', lw=0.5)
        # rect.set_zorder(100)
        return xmin_line, xmax_line, ymin_line, ymax_line

    def updater(self, ax, artists, bounds):
        xmin_line, xmax_line, ymin_line, ymax_line = artists
        xmin, xmax, ymin, ymax = bounds

        xmin_line.set_xdata(xmin)
        xmax_line.set_xdata(xmax)

        ymin_line.set_ydata(ymin)
        ymax_line.set_ydata(ymax)

    def update_preview(self, event):
        if self.updated_canvas.is_in_draw_mode():
            try:
                event = Canvas.map_imag_canvas(event)
                bounds = self.current_brush.get_bounds(event, self.updated_canvas.imag)
                if self.bounds_changed(bounds):
                    self.refresh(bounds=bounds)
                self.last_bounds = bounds
            except (TypeError, ValueError) as e:
                self.hide()
                pass
        else:
            self.hide()
            # self.updated_canvas.refresh()

    def bounds_changed(self, new_bounds):
        if self.last_bounds is not None:
            xmin, xmax, ymin, ymax = self.last_bounds
            new_xmin, new_xmax, new_ymin, new_ymax = new_bounds

            if xmin == new_xmin and ymin == new_ymin and xmax == new_xmax and ymax == new_ymax:
                return False
            else:
                return True
        else:
            return True

    def on_figure_leave(self, source):
        self.hide()
        self.updated_canvas.refresh()

    def on_figure_enter(self, source):
        pass


class WaveTablePreview(UpdatedGraphPlot):
    def __init__(self, wav_table, *args, **kwargs):
        UpdatedGraphPlot.__init__(self, *args, **kwargs)
        self.wav_table = wav_table
        self.set_new(self.wav_table, self.draw_wav_table, self.update_wav_table)
        self.refresh()

        self.toolbar.canvas.mpl_disconnect(self.toolbar._idDrag)

    def draw_wav_table(self, ax, data):
        coords, smoothed_coords = data
        ys, xs = coords
        line, = ax.plot(xs, ys)
        ax.axis('off')
        ax.set_xlim([min(xs), max(xs)])
        return line

    def update_wav_table(self, ax, line, data):
        coords, smoothed_coords = data
        ys, xs = coords
        line.set_ydata(ys)
        return line

    def refresh_wav_table(self, *args, **kwargs):
        self.refresh_data_obj(self.wav_table, *args, **kwargs)