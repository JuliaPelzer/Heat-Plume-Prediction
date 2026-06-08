from code.utils import logging as log  # noqa: F401

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# shifted hot colormap to better capture details
def new_cmap(colors, nodes, name: str = None):
    nodes[...] -= nodes[0]
    nodes /= nodes[-1]
    # log.info(nodes)

    if name:
        try:
            my_cmap = LinearSegmentedColormap.from_list(name, list(zip(nodes, colors, strict=True)))
            mpl.colormaps.register(cmap=my_cmap)
        except Exception:
            my_cmap = LinearSegmentedColormap.from_list("dummy", list(zip(nodes, colors, strict=True)))
            log.info("Already defined")


if True:
    # for temperature lower from 5.6 - 15.6
    mid = 10.6
    smax = 5.0

    s2 = 1.5  # tuneable
    s1 = 0.75  # tuneable
    s0 = 0.25  # tuneable
    nodes = np.array([mid - smax, mid - s2, mid - s1, mid - s0, mid, mid + s0, mid + s1, mid + s2, mid + smax])

    name = "jp_temperature_bidirectional"
    colors = ["#313695", "#4575B4", "#74ADD1", "#E0F3F8", "#FFFFFF", "#FEE090", "#FDAE61", "#F46D43", "#A50026"]
    new_cmap(colors, nodes, name)

    name = "jp_temperature_bidirectional_dark"
    colors = ["#80F3FF", "#00B4D8", "#0077B6", "#032030", "#000000", "#2E0505", "#C41E3A", "#FF6B00", "#FFD700"]
    new_cmap(colors, nodes, name)

if True:
    # for temperature lower from 10.6 - 15.6
    nodes = np.array([10.6, 11.7, 12.0, 13.5, 15.6])

    name = "jp_temperature_upperlinear"
    colors = ["white", "darkblue", "darkred", "orange", "yellow"]
    new_cmap(colors, nodes, name)

    name = "jp_temperature_upperlinear_dark"
    colors = ["black", "darkblue", "darkred", "orange", "yellow"]
    new_cmap(colors, nodes, name)

if True:
    # for temperature lower from 5.6 - 10.6
    nodes = np.array([5.6, 6.7, 7.0, 8.5, 10.6])

    name = "jp_temperature_lowerlinear"
    colors = ["yellow", "orange", "darkred", "darkblue", "white"]
    new_cmap(colors, nodes, name)

    # dark mode
    name = "jp_temperature_lowerlinear_dark"
    colors = ["yellow", "orange", "darkred", "darkblue", "black"]
    new_cmap(colors, nodes, name)

if True:
    # shifted hot colormap to better capture details
    name = "jp_linear"
    colors = ["white", "darkblue", "darkred", "orange", "yellow"]
    nodes = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    new_cmap(colors, nodes, name)

    # dark mode
    name = "jp_linear_dark"
    colors = ["black", "darkblue", "darkred", "orange", "yellow"]
    nodes = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    new_cmap(colors, nodes, name)
