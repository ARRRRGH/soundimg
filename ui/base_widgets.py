import run_time.config as config
import run_time.py_utils as py_utils
import data_structures.data_utils as data_utils
import ui.gtk_utils as gtk_utils

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GObject

import os
import pickle as pkl


class CollapseHeader(Gtk.EventBox):
    __gsignals__ = {"toggled": (GObject.SignalFlags.RUN_LAST, None, [GObject.TYPE_PYOBJECT])}

    def __init__(self, title_label=None):
        Gtk.EventBox.__init__(self)

        bar = Gtk.Box()

        # fixed = Gtk.Fixed()
        bar.set_size_request(height=config.section_header_height, width=-1)
        # fixed.add(bar)

        self.add(bar)

        if title_label is not None:
            title_label.set_valign(Gtk.Align.CENTER)
            bar.pack_start(title_label, False, False, config.section_hborder)

        self._active = False
        self.connect('button-press-event', self.clicked)

        context = self.get_style_context()
        context.add_class('collapse-header')

        self.show_all()

    def get_active(self):
        return self._active

    def set_active(self, bool):
        self._active = bool

    def clicked(self, *args, **kwargs):
        self._active = not self._active
        self.emit('toggled', self)


class CollapseSection(Gtk.VBox):
    def __init__(self, label, section):
        Gtk.VBox.__init__(self)
        self.header = CollapseHeader(label)
        self.section = section

        self.add(self.header)
        self.pack_start(self.section, True, True, config.section_hborder)

        self.header.connect('toggled', self.toggle)
        self.open()

        context = self.get_style_context()
        context.add_class('collapse-section')

        self.show()

    def collapse(self):
        if self.header.get_active():
            self.set_active(False)
        self.section.hide()

    def open(self):
        if not self.header.get_active():
            self.header.set_active(True)
        self.section.show()

    def toggle(self, *args, **kwargs):
        if self.header.get_active():
            self.open()
        else:
            self.collapse()


class CollapsingSectionWindow(Gtk.VBox):
    def __init__(self, *args, **kwargs):
        Gtk.VBox.__init__(self)

        self.scrolled_window = FocusScrolledWindow(*args, **kwargs)
        self.pack_start(self.scrolled_window, True, True, 0)

        self.children = []

        context = self.get_style_context()
        context.add_class('notebook')

    def add_child(self, child, border, with_header=True, title_label=None, *args, **kwargs):
        if with_header:
            child = CollapseSection(title_label, child)
        self.children.append(child)
        self.scrolled_window.add_child(child, border, *args, **kwargs)

    def remove_child(self, child):
        self.children.remove(child)
        self.scrolled_window.remove_child(child)

    def collapse_children(self):
        for child in self.children:
            child.collapse()

    def open_children(self):
        for child in self.children:
            child.open()


class FocusScrolledWindow(Gtk.ScrolledWindow):
    """
    Get a gtk.ScrolledWindow which contains a gtk.Viewport.
    Attach event handlers which will scroll it to show the focused widget.
    """

    def __init__(self, min_dims=None, max_dims=None, propagate_natural_height=False, propagate_natural_width=False):
        Gtk.ScrolledWindow.__init__(self)
        self.content_area = Gtk.VBox()
        self.add(self.content_area)

        self.vadj = self.get_vadjustment()
        self.vadj.set_lower(0)
        self.vadj.set_upper(1)
        self.vadj.set_value(0)
        self.vadj.set_page_size(0)

        if min_dims is not None:
            self.set_min_content_width(min_dims[1])
            self.set_min_content_height(min_dims[0])

        if max_dims is not None:
            self.set_max_content_width(max_dims[1])
            self.set_max_content_height(max_dims[0])

        # self.content_area.connect('draw', self.redraw)
        self.set_propagate_natural_height(propagate_natural_height)
        self.set_propagate_natural_width(propagate_natural_width)

    def is_child(self, widget):
        for child in self.content_area.get_children():
            if child is widget:
                return True
        return False

    def scroll_to_child(self, child):
        GObject.idle_add(self.adapt_viewport, child)

    def adapt_viewport(self, widget):
        """Scroll the viewport if needed to see the current focused widget"""

        if not self.is_child(widget):
            return

        _wleft, wtop = widget.translate_coordinates(self.content_area, 0, 0)
        wbottom = wtop + widget.get_allocation().height

        top = self.vadj.get_value()
        page_size = self.vadj.get_page_size()
        bottom = top + page_size

        if wtop < top:
            self.vadj.set_value(wtop)
        elif wbottom > bottom:
            self.vadj.set_value(wbottom - page_size)

    def add_child(self, child, border, *args, **kwargs):
        self.content_area.pack_start(child, False, False, 0)
        child.set_margin_bottom(border)

    def remove_child(self, child):
        self.content_area.remove(child)

    def redraw(self, source, cr):
        new_size = self.content_area.get_preferred_size()[0]
        self.content_area.set_size_request(new_size.width, new_size.height)
        self.show()


class StoreComboBox(Gtk.Box):
    def __init__(self, store, info_button=False, import_button=True, *args, **kwargs):
        Gtk.Box.__init__(self, *args, **kwargs)

        self.combo = Gtk.ComboBox()
        self.pack_start(self.combo, False, False, 0)

        self.import_button = Gtk.Button('...')

        self.import_button.connect('clicked', self.on_import_button_clicked)
        if import_button:
            self.show_import_button()

        if info_button:
            self.settings_button = Gtk.Button()
            self.settings_button.connect('clicked', self.on_info_button_clicked)
            self.pack_start(self.settings_button, False, False, config.default_hborder)

        self.store = store
        self.set_store()

        self.show_all()

    def hide_import_button(self):
        self.remove(self.import_button)

    def show_import_button(self):
        self.pack_start(self.import_button, False, False, config.default_hborder)

    def get_active_value(self):
        active_iter = self.combo.get_active_iter()
        if active_iter is not None:
            model = self.get_model()
            key = model[active_iter][0]
            return self.store.dict[key]
        else:
            return None

    def set_store(self):
        self.combo.set_model(self.store.gtk_store)
        renderer_text = Gtk.CellRendererText()
        self.combo.pack_start(renderer_text, True)
        self.combo.add_attribute(renderer_text, "text", 0)

    def replace(self, *args, **kwargs):
        self.store.replace(*args, **kwargs)
        self.set_store()

    def on_import_button_clicked(self, source):
        dialog = StoreDialog(self.store, action=Gtk.FileChooserAction.OPEN)
        response = dialog.run()
        dialog.destroy()

    def connect(self, *args, **kwargs):
        self.combo.connect(*args, **kwargs)

    def get_model(self):
        return self.combo.get_model()

    def on_info_button_clicked(self, source):
        try:
            self.get_active_value().show_settings_dialog()
        except AttributeError:
            pass


class StoreDialog(Gtk.VBox):
    def __init__(self, store, key=None, item=None, action=None):
        Gtk.VBox.__init__(self)
        self.store = store
        self.key = key
        self.item = item
        self.action = action

        preview_widget = _PreviewWidget()

        self.file_chooser = Gtk.FileChooserWidget()
        self.file_chooser.set_action(action)
        self.file_chooser.set_do_overwrite_confirmation(False)
        self.file_chooser.set_current_folder(self.store.storage_dir)

        self.file_chooser.set_preview_widget(preview_widget)
        self.file_chooser.connect('update-preview', preview_widget.update)

        self.add(self.file_chooser)

    def run(self):
        if self.action == Gtk.FileChooserAction.SAVE:
            kwargs = {'ok':False, 'cancel':True, 'save':True}
        elif self.action == Gtk.FileChooserAction.OPEN:
            kwargs = {'ok':True, 'cancel':True}
        else:
            raise NotImplementedError

        dialog = ResponseDialog('Store Editor', config.main_window, self, **kwargs)
        response = dialog.run()

        if response == Gtk.ResponseType.CANCEL:
            pass
        elif response == Gtk.ResponseType.ACCEPT and self.action == Gtk.FileChooserAction.SAVE:
            self.save()
        elif response == Gtk.ResponseType.OK and self.action == Gtk.FileChooserAction.OPEN:
            self.import_to_store()

        dialog.destroy()
        return response

    def save(self):
        name = self.file_chooser.get_current_name()

        # if name is chosen from Gtk generated suggestions
        if any(name.endswith(e) for e in config.exts):
            name = os.path.splitext(name)[0]

        if self.key is not None:
            item = self.store.dict.get_inv_key(self.key)
        elif self.item is not None:
            item = self.item
        else:
             raise ValueError('item or key must be given')

        self.store.save(item=item, name=name, parent=None)

    def import_to_store(self):
        path = py_utils.uri_to_path(self.file_chooser.get_uri())
        name, ext = os.path.splitext(os.path.basename(path))
        ext = ext[1:]

        if ext == self.store.ext:
            with open(path, 'rb') as f:
                item = pkl.load(f)
            self.store.add_to_store(item, name)


class _PreviewWidget(Gtk.Box):
    def __init__(self):
        Gtk.Box.__init__(self)
        self.child = None

    def set(self, preview):
        if self.child is not None:
            self.remove(self.child)
        self.child = preview
        self.add(self.child)
        self.show_all()

    def update(self, file_chooser):
        try:
            file_uri = file_chooser.get_preview_uri()

            fil = os.path.basename(file_uri)
            name, ext = os.path.splitext(fil)

            if ext[1:] in config.exts:
                preview_widget = file_chooser.get_preview_widget()
                with open(file_uri[7:], 'rb') as f:
                    obj = pkl.load(f)
                preview = obj.get_preview(dims=config.dims_brush_preview)
                self.set(preview)
                file_chooser.set_preview_widget_active(True)
            else:
                file_chooser.set_preview_widget_active(False)
        except AttributeError:
            pass


class ResponseDialog(Gtk.Dialog):
    def __init__(self, name, parent, info_widget, cancel=True, ok=True, save=False, remove=False, hide_on_destroy=False,
                 modal=True):

        button_data = []
        if ok:
            button_data += [Gtk.STOCK_OK, Gtk.ResponseType.OK]
        if cancel:
            button_data += [Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL]
        if save:
            button_data += [Gtk.STOCK_SAVE, Gtk.ResponseType.ACCEPT]
        if remove:
            button_data += [Gtk.STOCK_REMOVE, Gtk.ResponseType.REJECT]
        button_data = tuple(button_data)

        Gtk.Dialog.__init__(self, name, parent, 0, button_data)

        box = self.get_content_area()
        box.add(info_widget)
        box.set_spacing(config.default_hborder)

        if hide_on_destroy:
            self.connect('destroy', lambda source: self.hide())
        self.set_modal(modal)

        self.show_all()


class SettingsGrid(Gtk.Grid):
    def __init__(self, settings, obj):
        Gtk.Grid.__init__(self)
        self.obj = obj
        self.ranges = {}

        self.set_row_spacing(config.default_rowspacing)
        self.set_column_spacing(config.default_colspacing)

        self.settings = settings
        self.sources = {}

        for i, (var_name, var, typ, mode, mode_kwargs, width, height) in enumerate(settings):
            self.attach(Gtk.Label(var_name), 0, i, 1, 1)

            if mode == 'entry':
                entry = Gtk.Entry()
                self.sources[entry] = var, typ
                entry.connect('changed', self.on_entry_changed)
                self.attach(entry, 1, i, width, height)
                entry.set_text(str(getattr(self.obj, var)))

                if mode_kwargs is not None:
                    if 'range' in mode_kwargs.keys():
                        self.ranges[var] = mode_kwargs['range']

            elif mode == 'scale':
                scale = Gtk.Scale()
                self.sources[scale] = var, typ
                scale.connect('value-changed', self.on_scale_changed)
                self.attach(scale, 1, i, width, height)
                scale.set_value(getattr(self.obj, var))

                if mode_kwargs is not None:
                    if 'range' in mode_kwargs.keys():
                        scale.set_range(*mode_kwargs['range'])
                    if 'increments' in mode_kwargs.keys():
                        scale.set_increments(*mode_kwargs['increments'])

            elif mode == 'combo':
                store = gtk_utils.BaseStore(typ=str)
                combo = StoreComboBox(store, import_button=False)
                self.sources[combo] = var, typ
                combo.connect('changed', self.on_combo_changed)
                self.attach(combo, 1, i, width, height)
                if mode_kwargs is not None:
                    if 'store' in mode_kwargs:
                        for item in mode_kwargs['store']:
                            try:
                                item, name = item
                            except:
                                item, name = item, str(item)

                            store.add_to_store(item, name)

            elif mode == 'switch':
                switch = Gtk.Switch()
                self.sources[switch] = var, typ
                switch.connect("notify::active", self.on_switch_activated)
                switch.set_active(True)
                self.attach(switch, 1, i, width, height)

    def on_entry_changed(self, source):
        var, typ = self.sources[source]
        new = gtk_utils.read_entry(source, typ)

        if var in self.ranges.keys():
            new = max(self.ranges[var][0], min(self.ranges[var][1], new))
            source.set_text(str(new))

        setattr(self.obj, var, new)

    def on_scale_changed(self, source):
        var, typ = self.sources[source]
        new = typ(source.get_value())

        setattr(self.obj, var, new)

    def on_combo_changed(self, source, data=None):
        var, typ = self.sources[source]
        new = source.get_active_value()

        setattr(self.obj, var, new)

    def on_switch_activated(self, source, data=None):
        var, typ = self.sources[source]
        new = source.get_active()

        setattr(self.obj, var, new)



class SettingsDialog(Gtk.Box):
    def __init__(self, settings, obj, name, parent):
        Gtk.Box.__init__(self)
        self.grid = SettingsGrid(settings, obj)
        self.dialog = ResponseDialog(name, parent, self.grid, cancel=False, ok=True, hide_on_destroy=True, modal=False)
        self.dialog.hide()

    def run(self):
        response = self.dialog.run()
        self.dialog.hide()
        return response


class EclipseWindow(Gtk.Box):
    def __init__(self, child1, child2, hide_button_parent, dims=None):
        Gtk.Box.__init__(self)

        self.child1 = child1
        # self.add(child1)
        self.child2 = child2
        self.add(child2)

        hide_button = Gtk.ToggleButton('x')
        hide_button.connect('toggled', self.on_hide_button_toggled)
        hide_button_parent.pack_end(hide_button, False, False, 0)

        self.on_hide_button_toggled(hide_button)

        if dims is not None:
            self.set_size_request(dims[1], dims[0])

    def set_child1_active(self):
        self.child2.eclipse_remove(self)
        self.child1.eclipse_show(self)

    def set_child2_active(self):
        self.child1.eclipse_remove(self)
        self.child2.eclipse_show(self)

    def on_hide_button_toggled(self, source):
        if source.get_active():
            self.set_child2_active()
        else:
            self.set_child1_active()


class EclipseChild(object):

    def eclipse_show(self, parent):
        parent.pack_start(self, True, True, 0)
        self.show_all()

    def eclipse_remove(self, parent):
        parent.remove(self)


class Notebook(Gtk.Notebook):
    def __init__(self):
        Gtk.Notebook.__init__(self)
        self.page_nrs = data_utils.UniqueDict()
        self._nr_pages = 0

    def append_page(self, child, child_key=None, *args, **kwargs):
        if child_key is None:
            child_key = child
        super(Notebook, self).append_page(child, *args, **kwargs)
        self.page_nrs[child_key] = self._nr_pages
        self._nr_pages += 1

    def remove_page(self, child_key_or_ind):
        child_key, removed_page_nr = self.page_nrs.get_key(child_key_or_ind), self.page_nrs.get_inv_key(child_key_or_ind)
        del self.page_nrs[child_key]
        self._remove(removed_page_nr)

        return child_key

    def _remove(self, ind):
        super(Notebook, self).remove_page(ind)

        # update page numbers
        for val in sorted(self.page_nrs.values()):
            if val > ind:
                self.page_nrs.change_key(val, val - 1)
        self._nr_pages -= 1

    def remove_current_page(self):
        ind = self.get_current_page()
        if ind != -1:
            return self.remove_page(ind)