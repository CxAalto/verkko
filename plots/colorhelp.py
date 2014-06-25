import colorsys
import numpy as np

"""
This modules contains helper functions for obtaining sets of distinct colors.
More color-related functions are welcome.
"""


def get_distinct_colors(num_colors):
    """
    Helper function for getting an arbitrary number of distinct colors.
    The algorithm is purely heuristical.
    """
    colorList = []
    hueMax = 360.
    for j, i in enumerate(np.linspace(0., hueMax, num_colors)):
        hue = i / 360.
        lightness = (38 + j % 3 * 17) / 100.
        saturation = (80 + j % 2 * 20) / 100.
        colorList.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return np.array(colorList)


"""
See the origin of colorbrewer palettes at:
http://colorbrewer2.org/
"""

CBREWER11 = np.array([
                     (166, 206, 227),
                    (31, 120, 180),
                    (178, 223, 138),
                    (51, 160, 44),
                    (251, 154, 153),
                    (227, 26, 28),
                    (253, 191, 111),
                    (255, 127, 0),
                    (202, 178, 214),
                    (106, 61, 154),
                    (255, 255, 153)]
                     ) / 255.

CBREWER09 = np.array([[228, 26, 28],
                      [247, 129, 191],
                      [77, 175, 74],
                      [255, 255, 51],
                      [255, 127, 0],
                      [152, 78, 163],
                      [55, 126, 184],
                      [166, 86, 40],
                      [153, 153, 153]])
