import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# shifted hot colormap to better capture details
def new_cmap(colors, nodes, name:str=None):
    nodes[...] -= nodes[0]
    nodes /= nodes[-1]
    # print(nodes)

    if name:
        try:
            my_cmap = LinearSegmentedColormap.from_list(name, list(zip(nodes, colors)))
            mpl.colormaps.register(cmap=my_cmap)
        except:
            my_cmap = LinearSegmentedColormap.from_list("dummy", list(zip(nodes, colors)))
            print("Already defined")

# shifted hot colormap to better capture details
name = "jp_wbRow_neon"
colors = ["white", "blue", "darkred", "orange", "white"]
nodes = np.array([10.6, 11.7, 12., 13.5, 15.6])
new_cmap(colors, nodes, name)