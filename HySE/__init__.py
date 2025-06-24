import numpy as np
import cv2
import os
from datetime import datetime
from scipy.signal import savgol_filter, find_peaks
import matplotlib
from matplotlib import pyplot as plt
import imageio
# import matplotlib.colors as colors
# import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable



import pkgutil
import importlib
import inspect

__all__ = []

# Discover all .py files in the current package (excluding __init__.py and this file)
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")
    # Pull in public functions
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("_"):
            globals()[name] = obj
            __all__.append(name)
