from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

def _aligned_colorbar(*args, **kwargs):
    cax = make_axes_locatable(plt.gca()).append_axes(
        "right", size=0.3, pad=0.05)
    plt.colorbar(*args, cax=cax, **kwargs)
