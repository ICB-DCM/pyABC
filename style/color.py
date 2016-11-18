from itertools import starmap
from collections import UserDict
import matplotlib as mpl
import seaborn as sns


def rgb_to_rgb(r1, r2, r3):
    return r1 / 255, r2 / 255, r3 / 255


def rgb_to_hex(rgb):
    colors = [str(hex(int(channel*255)))[2:] for channel in rgb]
    colors = [col if len(col) == 2 else "0" + col for col in colors ]
    return "#" + "".join(colors)



# colors from tango icon theme: chocolate and plum
matrix_colors = [(0xe9/255, 0xb9/255, 0x6e/255),
                 (0, 0, 0),
                 (0xad/255, 0x7f/255, 0xa8/255)]


def mimic_alpha(color: tuple, alpha: float, background_color=(1,1,1)):
    return tuple(c * alpha + (1-alpha) * bc for c, bc in zip(color, background_color))


wang_nature_colorblind = [
    (0, 0, 0),  # Black
    (230, 159, 0),  # Orange
    (86, 180, 233),  # Sky blue
    (  0, 158, 115),     # Bluish green
    (240, 228, 66),  # Yellow
    (  0, 114, 178),     # Blue
    (213,  94,   0),      # Vermillon
    (204, 121, 167),   # Reddish purple
]


wang_nature_colorblind = list(starmap(rgb_to_rgb, wang_nature_colorblind))


_color  = {"thing":   (  0,   0,   0),         # Black
          "other_thing": (  0, 158, 115),     # Bluish green
          "gray": (100, 100, 100),    # Gray
          "black": (  0,   0,   0),
          "light_gray": (220, 220, 220)
          }


sns.set_palette(wang_nature_colorblind)


class ColorDictionary(UserDict):
    def __missing__(self, key):
        return self["gray"]

color = ColorDictionary({key: rgb_to_rgb(*value) for key, value in _color.items()})

cmap = mpl.cm.viridis