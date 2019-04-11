import data_structures.data_utils as data_utils
import run_time.py_utils as py_utils
import run_time.config as config

# import cv2
import numpy as np
import copy


class Image(object):
    """ Image is the actual container for images.

    Image holds and manages the image in form of a np.array. It is aware of the changes applied to the array until
    set_state(0, range_) is called."""

    def __init__(self, dims=None, img_path=None, img=None):
        # gtk_utils.GtkListener.__init__(self)

        # if img_path is not None:
        #     self.img = cv2.imread(img_path)
        if img is not None:
            self.img = img.astype(np.uint8)
        else:
            if dims is None:
                dims = (100, 100)
            height, width = dims
            self.img = np.zeros((height, width, 3), dtype=np.uint8)

        # for ease of calculation (brush)
        self.grid = None
        self.state_col = None
        self.fixed_state_px = {}
        self.update_img_data()

        self.name = 'Image'
        # self.listen_to('name')

    @property
    def shape(self):
        return tuple(list(self.img.shape)[0:2])

    def set_fixed_state_pxs(self, key):
        for px, vals in self.fixed_state_px.items():
            pop = False
            if isinstance(key, int):
                pop = px == key
            elif isinstance(key, slice):
                pop = key.start % config.params.image_width <= px < key.stop % config.params.image_width
            elif key is None:
                pop = True

            if pop:
                try:
                    val = vals.pop()
                    self.state_col[px] = val
                except IndexError:
                    del self.fixed_state_px[px]
                except TypeError:
                    pass

    def update_img_data(self):
        self.grid = np.indices(self.img.shape)
        self.state_col = data_utils.LoopArray(np.ones(self.shape[1], np.int), np.int, shared=False, modes=['wrap'])

    def set_state_col(self, val_int, affected_range=None):
        if affected_range is None:
            self.state_col[:] = np.ones(self.shape[1], np.int) * val_int
        else:
            self.state_col[affected_range] = 1 * val_int
        # self.set_fixed_state_pxs(affected_range)

    def set_img(self, mask, changed_img):
        self.img[:, :, :] = changed_img

        changed_cols = np.where(mask)[1]
        ch_min = np.min(changed_cols)
        ch_max = np.max(changed_cols)
        changed_cols_range = slice(ch_min, ch_max + 1)

        if changed_cols_range is not None:
            self.state_col[changed_cols_range] = 1

        return None

    def reset(self, height=None, width=None):
        shape_changed = False

        new_height = self.shape[0]
        if height is not None:
            shape_changed = True
            new_height = height

        new_width = self.shape[1]
        if width is not None:
            shape_changed = True
            new_width = width

        self.img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        if shape_changed:
           self.update_img_data()

        self.set_state_col(1)

    def is_changed_in_range(self, key):
        return np.sum(self.state_col[key])

    def get_next_changed_px(self, current_index):
        return py_utils.get_next_value_in_arr(self.state_col.arr, 1, 1, current_index, wrap=True)

    def extend(self, new_shape):
        diff = (new_shape[0] - self.shape[0], new_shape[1] - self.shape[1])
        if diff != (0, 0) and diff[0] >= 0 and diff[1] >= 0:
            tmp = np.zeros((new_shape[0], new_shape[1], 3), dtype=np.uint8)
            y_axis_lower = int(np.floor(diff[0] / 2.))
            y_axis_upper = int(np.ceil(diff[0] / 2.))

            x_axis_lower = int(np.floor(diff[1] / 2.))
            x_axis_upper = int(np.ceil(diff[1] / 2.))

            tmp[y_axis_lower:self.img.shape[0] + y_axis_upper, x_axis_lower:self.img.shape[1] + x_axis_upper, :] = self.img

            self.img = tmp
            self.update_img_data()

    def get_preview(self, dims):
        import data_structures.imaging as imaging
        canvas = imaging.Canvas(imag=self, dims_canvas=dims)
        canvas.refresh()
        return canvas.canvas

    def duplicate(self):
        return copy.deepcopy(self)


class PlayableImage(object):
    """ Interface of an Image to a TrackGenerator.

    PlayableImage provides the methods necessary for TrackGenerator to operate on an Image.
    """

    def __init__(self, track, *args, **kwargs):
        self.track = track
        self.image = track.image

        # extend image's members to support wrapping
        self.img = data_utils.LoopArray(self.image.img, np.uint8, shared=False, modes=['raise', 'wrap', 'raise'])

        # add members needed for sound synthesis
        self.init_conditions = data_utils.InitialConditions(self.shape, self.track)

    @property
    def shape(self):
        return self.image.shape

    @property
    def grid(self):
        return self.image.grid

    def gen_sound(self, synthesizer, px_range):
        i_conditions, new_start = self.init_conditions.get_last(px_range)
        if new_start != px_range.start:
            px_range = slice(new_start, px_range.stop)
        data, sample_range, f_conditions = synthesizer.generate_sound_from_image(self.img, px_range, i_conditions)

        if self.init_conditions.is_currently_reset():
            self.delay_px_recalculation(px_range.start)

        next = px_range.stop % self.shape[1]
        self.init_conditions.set(conditions=f_conditions, where=next)

        # set col state to 0 == 'unchanged/already played'
        self.image.set_state_col(0, px_range)

        return data, sample_range

    def delay_px_recalculation(self, px):
        self.image.fixed_state_px[px] = [1, 0]

    def reset(self):
        self.init_conditions.reset()

    def set(self, *args, **kwargs):
        self.init_conditions.set(*args, **kwargs)