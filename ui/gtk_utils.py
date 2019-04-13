import data_structures.data_utils as data_utils

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, GObject, Gdk

import pickle as pkl
import os


class _IdleObject(GObject.GObject):
    """
    Override GObject.GObject to always emit signals in the main thread
    by emitting on an idle handler
    """

    # @trace
    def __init__(self):
        GObject.GObject.__init__(self)

    # @trace
    def emit(self, *args):
        GObject.idle_add(GObject.GObject.emit, self, *args)


class GtkListener(_IdleObject):
    __listened_members__ = []
    __gsignals__ = {'update': (GObject.SignalFlags.RUN_LAST, None, [GObject.TYPE_STRING, GObject.TYPE_PYOBJECT])}

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if key in self.__listened_members__:
            self.emit('update', key, self.__dict__[key])

    def listen_to(self, member):
        self.__listened_members__.append(member)


class UpdatedGtkClass(object):
    def _update(self, source, member, updated_value):
        if member in self.__listened_members__ and source is not self:
            self.on_member_update(self, member, updated_value)

    def set_on_member_update(self, callable_update):
        self.on_member_update = callable_update

    def add_listened_member(self, member):
        self.__listened_members__.append(member)

    def on_member_update(self, source, this_widget, member, updated_value):
        pass


def get_updated_gtk_class(name, base_gtk_class=Gtk.Label, base_class=object):

    def __init__(self, gtk_listener, listened_members, *args, **kwargs):
        base_gtk_class.__init__(self)
        base_class.__init__(self, *args, **kwargs)

        self.__listened_members__ = []
        for member in listened_members:
            self.__listened_members__.append(member)

        self.gtk_listener = gtk_listener
        self.gtk_listener.connect('update', self._update)

    new_class = type(name, (UpdatedGtkClass, base_gtk_class, base_class), {"__init__": __init__})
    return new_class



UpdatedLabel = get_updated_gtk_class('UpdatedLabel', Gtk.Label)


UpdatedEntry = get_updated_gtk_class('UpdatedEntry', Gtk.Entry)


class BaseStore(_IdleObject):
    __gsignals__ = {'error': (GObject.SignalFlags.RUN_LAST, None, [GObject.TYPE_STRING]),
                    'info': (GObject.SignalFlags.RUN_LAST, None, [GObject.TYPE_STRING])}

    def __init__(self, typ, store=None, protected_names=None):
        _IdleObject.__init__(self)

        self.typ = typ

        self._protected_names = protected_names
        if protected_names is None:
            self._protected_names = []
        self._protected_names.append('')

        self.gtk_store = Gtk.ListStore(str)
        self.dict = data_utils.UniqueDict()
        self.iter_name = {}

        if store is not None:
            for item, name in store:
                self.add_to_store(item, name)

    @property
    def length(self):
        return len(self.dict)

    def append_protected_names(self, protected_names):
        self._protected_names += protected_names

    def add_to_store(self, item, name, protected=False):
        if isinstance(item, self.typ):
            iter = self.gtk_store.append([name])
            self.iter_name[name] = iter
            self.dict[name] = item
            if protected:
                self._protected_names.append(name)
        else:
            raise TypeError(str(item) + ' is not of type '+ str(self.typ))

    def __getitem__(self, key):
        return self.dict[key]

    def __setitem__(self, name, item):
        self.add_to_store(item, name)

    def replace(self, item_or_name, new_item_or_name):
        item = self.dict.get_inv_key(item_or_name)
        name = self.dict.get_key(item_or_name)

        protected = False
        if name in self._protected_names:
            protected = True

        if self.dict.is_key(item_or_name):
            self.remove(name)
            self.add_to_store(item, new_item_or_name, protected=protected)
        elif self.dict.is_inv_key(item_or_name):
            self.remove(name)
            self.add_to_store(new_item_or_name, name, protected=protected)


class FileStore(BaseStore):
    def __init__(self, storage_dir, ext, typ, store=None, loading_dirs=None, protected_names=None):
        BaseStore.__init__(self, typ, store=store, protected_names=protected_names)

        if not issubclass(self.typ, data_utils.StoreInterface):
            raise Exception('type specified in kwarg typ must implement the StoreInterface')

        self.loading_dirs = []
        if loading_dirs is not None:
            self.loading_dirs = loading_dirs

        self.storage_dir = storage_dir
        if storage_dir is not None:
            self.loading_dirs.append(storage_dir)
        self.ext = ext

        self.load()

    def load(self):
        for dir in self.loading_dirs:
            for root, dirs, files in os.walk(dir):
                for file in files:
                    if file.endswith(self.ext):
                        try:
                            with open(os.path.join(root, file), 'rb') as f:
                                item = pkl.load(f)
                            name = os.path.splitext(file)[0]
                            self.add_to_store(item, name, False)
                        except EOFError as e:
                            print(e)
                            pass

    def save(self, item, name, parent=None):
        if name in self._protected_names:
            self.emit('error', name + ' is internal and can\'t be saved')
        else:
            out_name = self.get_path(name)
            item = item.duplicate()
            item.name = name
            if name in self.dict:
                if self.dict[item] == name:
                    self.replace_save(item, name)
                else:
                    dialog = Gtk.MessageDialog(parent, 0, Gtk.MessageType.INFO,
                                               Gtk.ButtonsType.OK_CANCEL,
                                               "Do you want to override '" + str(name) + "'?")
                    response = dialog.run()
                    if response == Gtk.ResponseType.OK:
                        self.replace_save(item, name)
                    else:
                        self.emit('error', "Didn't override" + str(name))
                    dialog.destroy()
            else:
                self.add_to_store(item, name)
                self.emit('info', 'Saved ' + self.dict[item] + ' to ' + name)

            with open(out_name, "wb") as f:
                pkl.dump(item, f)

    def replace_save(self, item, name):
        self.replace(self.dict[name], item)
        self.emit('info', 'Saved ' + name)

    def remove(self, item_or_name):
        name = self.dict.get_key(item_or_name)
        if name in self.dict and not name in self._protected_names:
            self.gtk_store.remove(self.iter_name[name])
            del self.dict[item_or_name]
            del self.iter_name[name]

            if name in self._protected_names:
                self._protected_names.remove(name)

            try:
                os.remove(self.get_path(item_or_name))
                self.emit('info', 'Removed ' + name + ' permanently')
            except:
                self.emit('error', 'Could not remove ' + name + ' permanently.')
                pass
        else:
            self.emit('error', name + ' is internal and can\'t be saved')

    def get_path(self, name):
        return os.path.join(self.storage_dir, name + '.' + self.ext)


class ListenedValue(GtkListener):
    def __init__(self, val):
        GtkListener.__init__(self)
        self.val = val
        self.listen_to('val')

    def get_value(self):
        return self.val


class SignalManager(object):
    def __init__(self):
        self._signals = []
        self._sources = {}
        self._connections = []

    def add_signal(self, name, source, key=None):
        if not (name, source) in self._signals:
            if key is None:
                key = source
            self._sources[key] = source
            self._signals.append((name, source))

    def connect(self, name, target_callable, specific_source=None):
        for n, source in self._signals:
            if n == name:
                if (specific_source is not None and source == specific_source) or specific_source is None:
                    id = source.connect(name, target_callable)
                    self._connections.append((name, source, target_callable, id))

    def clear_from_source(self, source):
        tmp = []
        for connection in self._connections:
            n, s, t, id = connection
            if s == source:
                GObject.signal_handler_disconnect(s, id)
            else:
                tmp.append(connection)
        self._connections = tmp
        self._signals = [(name, s) for name, s in self._signals if s != source]
        self._sources = {key: s for key, s in self._sources.items() if s != source}

    def clear_from_key(self, key):
        self.clear_from_source(self._sources[key])

    def make_emit(self, name, *args, **kwargs):
        for n, source in self._signals:
            if n == name:
                source.emit(*args, **kwargs)



def move_widget_btw_boxes(widget, parent, new_parent):
    parent.remove(widget)
    new_parent.add(widget)


def find_children_of_type(parent, typ, ret=[]):
    try:
        children = parent.get_children()
    except:
        return None

    for child in children:
        if type(child) == typ:
            ret.append(child)
        else:
            find_children_of_type(child, typ, ret)

    return ret


def on_destroy_remove_child(source, *args, **kwargs):
    for child in source.get_children():
        source.remove(child)
    source.destroy()


def set_text_buffer_on_update(this_widget, member, updated_value):
    this_widget.set_text(str(updated_value))


def read_entry(entry, cast=int):
    try:
        return cast(entry.get_text())
    except ValueError or TypeError:
        raise NotImplementedError


def pack_centered(parent, widget, pack_start_opts=None):
    hbox = Gtk.HBox()
    vbox = Gtk.VBox()
    hbox.pack_start(vbox, True, True, 0)
    vbox.pack_start(widget, True, True, 0)

    if pack_start_opts is None:
        pack_start_opts = [True, True, 0]
    parent.pack_start(hbox, *pack_start_opts)


class StoreTreeView(Gtk.TreeView):
    def __init__(self, store, is_editable=False, on_edit=None):
        Gtk.TreeView.__init__(self)

        # self.active_items = []
        # self.get_selection().connect('changed', self.update_active_items)

        self.set_model(store.gtk_store)

        self.renderer = Gtk.CellRendererText()

        self.on_editing_started_tmp_name = None
        self.on_edit = on_edit

        if is_editable:
            self.renderer.props.editable = is_editable
            self.renderer.connect('editing-started', self.on_cell_editing_started)
            self.renderer.connect('editing-canceled', self.on_cell_editing_cancel)

        column = Gtk.TreeViewColumn(store.name, self.renderer)
        column.add_attribute(self.renderer, "text", 0)

        self.append_column(column)
        self.show_all()

    def get_active_rows(self):
        model, treeiter = self.get_selection().get_selected()
        if treeiter is not None:
            ret = model[treeiter]
            if type(ret) != list:
                return [ret]
            return ret
        else:
            return None

    def get_active_treeiter(self):
        model, treeiter = self.get_selection().get_selected()
        return treeiter

    def on_cell_editing_started(self, cell_renderer_text, editable, path):
        self._on_editing_cancelled_tmp_text = editable.get_text()

    def on_cell_editing_cancel(self, cell_editable):
        new_text = cell_editable.get_text()
        self.on_edit(self._on_editing_cancelled_tmp_text, new_text)


def add_with_border(container, widget, px, color_string):
    color = Gdk.RGBA()
    color.parse(color_string)

    eb = Gtk.EventBox()
    eb.modify_bg(Gtk.StateType.NORMAL, color.to_color())

    color.parse('white')
    eb_bg = Gtk.EventBox()
    eb_bg.modify_bg(Gtk.StateType.NORMAL, color.to_color())

    if hasattr(px, '__iter__'):
        px_left, px_right, px_top, px_bottom = px
        eb_bg.set_margin_left(px_left)
        eb_bg.set_margin_right(px_right)
        eb_bg.set_margin_top(px_top)
        eb_bg.set_margin_bottom(px_bottom)
    else:
        eb_bg.set_border_width(px)

    eb_bg.add(widget)
    eb.add(eb_bg)
    container.add(eb)
    return eb
