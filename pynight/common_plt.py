import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np

def colormap_get(cmap_name, start=0.0, end=1.0, name='viridis', reverse_p=False):
    """
    Returns a new matplotlib colormap using a subset of an existing colormap

    :param cmap_name: The name of the existing matplotlib colormap
    :param start: The start ratio where the new colormap begins from the existing colormap
    :param end: The end ratio where the new colormap ends from the existing colormap
    :param name: The name for the new colormap
    :param reverse_p: A boolean indicating whether to reverse the colormap
    :return: The new matplotlib colormap
    """
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    colors = cmap(np.linspace(start, end, cmap.N))

    if reverse_p:
        colors = colors[::-1]

    return matplotlib.colors.LinearSegmentedColormap.from_list(name, colors)
