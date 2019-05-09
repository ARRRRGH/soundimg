import numpy as np


class Params:
    def __init__(self):
        self.output_sample_rate = 44100
        self.dtype = np.float32
        self.image_width = 100
        self.block_length = 3000
        self.channels = 2
        self.concert_a = 440

    @property
    def buffer_nr_points_displayed_per_px(self):
        return np.int(max(10, min(project.params.sound_gen_params.samples_per_px / 100., 100)))

    @property
    def buffer_nr_points_displayed(self):
        return params.image_width * self.buffer_nr_points_displayed_per_px


# run time parameters
params = Params()

# manager instances
thread_manager = None
signal_manager = None

# variables

# global widgets
project = None
player = None
brush_window = None
main_window = None
instrument_overview = None
freq_map_overview = None
buffer_overview = None

# global information
def get_time_px():
    played_samples = player.played_samples
    return player.get_absolute_px_from_sample(played_samples)


# appearance track pane
track_pane_info_bar_width = 15
track_pane_height_shrinked = 150
track_pane_height_opened = 300
track_pane_width = 700
track_pane_border = 3


# main window appearance
dims_main_player_canvas = (698, 700)
dims_amp_view = (730, 150)

dims_brush_preview = (100, 100)
dims_brush_preview_canvas = (400, 400)
dims_brush_window = (730, 400)

dims_wave_table_preview = (200, 700)
dims_track_limiter_editor = (200, 700)
dims_fader_editor = (200, 340)
dims_freq_map_editor = (200, 700)
dims_save_menu_preview = (500, 500)
dims_preview_image = (200, 200)
dims_init_track = (track_pane_height_opened + track_pane_info_bar_width + track_pane_border, track_pane_width)
start_menu_image_height = 300

section_border = 3
section_hborder = 4
default_hborder = 5
default_rowspacing = 3 * default_hborder
default_colspacing = 5 * default_hborder
section_rubberband = 0
section_header_height = 30


brush_preview_color = 'red'

# stores
brush_store = None
instrument_store = None
wave_tables_store = None
freq_mappings_store = None
modulation_store = None
brush_mode_store = None
track_image_store = None
standard_scale_store = None

# other settings
time_pointer_callback_time = 20

# pathes
icons_path = '/Users/jim/soundimg/gui/images/icons'
pycache_path = '/Users/jim/soundimg/synthesis/__pycache__'
glade_dir = '/Users/jim/soundimg/gui/glade'
project_file_name = '.ingrid_project'
exts = ('brush', 'instrument', 'wave_table', 'modulation')

FFMPEG_BIN = "/anaconda2/envs/soundimg2/bin/ffmpeg" # on Linux
