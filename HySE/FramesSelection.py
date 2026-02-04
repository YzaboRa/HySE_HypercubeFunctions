

"""

TEST

"""

import numpy as np
import cv2
import os
from datetime import datetime
from scipy.signal import savgol_filter, find_peaks
import matplotlib
from matplotlib import pyplot as plt
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import RectangleSelector
# import SimpleITK as sitk
import time
from tqdm import trange
import inspect

matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"


import HySE.UserTools
import HySE.Import
import HySE.ManipulateHypercube
import HySE.CoRegistrationTools
import HySE.ManualRegistration


PythonEnvironment = get_ipython().__class__.__name__

from ._optional import sitk as _sitk
from skimage.metrics import normalized_mutual_information as nmi 
from scipy.ndimage import gaussian_filter

from PIL import Image
from natsort import natsorted
import glob
import copy
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import tempfile

import SimpleITK as _sitk
# import matplotlib.colors as mcolors
# import matplotlib.cm as cm


from matplotlib.widgets import Slider, RadioButtons
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from skimage import transform
from skimage.draw import polygon


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons

OriginPosition = 'lower'

class FrameSelector:
	def __init__(self, hypercube, wavelength_labels=None):
		"""
		GUI for selecting usable frames in a Hypercube [Nsweep, Nwavelengths, Y, X].
		
		Parameters:
		-----------
		hypercube : np.ndarray
			4D array of shape [Nsweep, Nwavelengths, Y, X]
		wavelength_labels : list, optional
			List of labels for the wavelength dimension. 


		How to use:
		GUI = HySE.FrameSelector(HypercubeForRegistration)
		good_frames_mask, good_indices = GUI.get_results()

		where
		- good_frames_mask : boolean numpy array of size [Nsweeps, Nwavelengths].
			True = Keep frame, False = Ignore Frame

		- good_indices : numpy array of size [total_usable_frames, 2]
			contains for each usable frame [sweep_index, frame_index]
		"""
		self.cube = hypercube
		self.n_sweeps, self.n_wavs, self.h, self.w = hypercube.shape
		
		# Labels
		if wavelength_labels is None:
			self.wav_labels = [f"Frame {i}" for i in range(self.n_wavs)]
		else:
			self.wav_labels = wavelength_labels

		# State: Keep track of usable frames (Default: All True)
		self.good_frames = np.zeros((self.n_sweeps, self.n_wavs), dtype=bool)
		
		# Navigation State
		self.curr_sweep = 0
		self.curr_wav = 0
		self.internal_update = False # Flag to prevent callback loops
		
		# Visual State
		self.cmap_name = 'gray'
		# Calculate global percentiles for initial contrast limits
		flat_sample = hypercube[::max(1, self.n_sweeps//5)].flatten() # Subsample for speed
		self.vmin = np.percentile(flat_sample, 1)
		self.vmax = np.percentile(flat_sample, 99)
		self.global_min = np.min(hypercube)
		self.global_max = np.max(hypercube)

		# --- GUI Layout ---
		self.fig = plt.figure(figsize=(14, 9))
		
		# Image Area
		self.ax_img = self.fig.add_axes([0.05, 0.25, 0.55, 0.7])
		self.ax_img.set_axis_off()
		self.img_handle = None

		# --- Controls Area (Bottom) ---
		
		# Navigation Sliders
		ax_sweep = self.fig.add_axes([0.05, 0.12, 0.4, 0.03])
		ax_wav = self.fig.add_axes([0.05, 0.08, 0.4, 0.03])
		
		self.slider_sweep = Slider(ax_sweep, 'Sweep', 0, self.n_sweeps - 1, valinit=0, valstep=1)
		self.slider_wav = Slider(ax_wav, 'Frame', 0, self.n_wavs - 1, valinit=0, valstep=1)
		
		# Keep/Ignore Checkbox
		ax_check = self.fig.add_axes([0.50, 0.1, 0.1, 0.08])
		self.chk_keep = CheckButtons(ax_check, ['Keep\nFrame'], [True])
		
		# --- Visualization Controls (Right Side) ---
		
		# Colormap Selection
		ax_cmap = self.fig.add_axes([0.65, 0.8, 0.12, 0.12])
		ax_cmap.set_title("Colormap")
		self.radio_cmap = RadioButtons(ax_cmap, ('Grayscale', 'Magma', 'Viridis'), active=0)
		
		# Contrast Sliders (Min/Max)
		ax_vmin = self.fig.add_axes([0.65, 0.70, 0.12, 0.03])
		ax_vmax = self.fig.add_axes([0.65, 0.65, 0.12, 0.03])
		
		self.slider_vmin = Slider(ax_vmin, 'Min', self.global_min, self.global_max, valinit=self.vmin)
		self.slider_vmax = Slider(ax_vmax, 'Max', self.global_min, self.global_max, valinit=self.vmax)
		
		# Stats Display
		ax_stats = self.fig.add_axes([0.82, 0.25, 0.15, 0.7])
		ax_stats.axis('off')
		ax_stats.set_title("Usable Frames Count", fontweight='bold')
		self.txt_stats = ax_stats.text(0, 0.98, "", va='top', ha='left', fontsize=9, family='monospace')
		
		# Done Button
		ax_done = self.fig.add_axes([0.8, 0.05, 0.15, 0.05])
		self.btn_done = Button(ax_done, 'Finish Selection')
		
		# --- Connections ---
		self.slider_sweep.on_changed(self.on_nav_change)
		self.slider_wav.on_changed(self.on_nav_change)
		self.chk_keep.on_clicked(self.on_keep_toggle)
		self.radio_cmap.on_clicked(self.on_cmap_change)
		self.slider_vmin.on_changed(self.on_contrast_change)
		self.slider_vmax.on_changed(self.on_contrast_change)
		self.btn_done.on_clicked(self.finish)
		
		# Initial Draw
		self.update_image()
		self.update_stats()
		plt.show()

	def on_nav_change(self, val):
		"""Handle slider movements for Sweep or Wavelength."""
		self.curr_sweep = int(self.slider_sweep.val)
		self.curr_wav = int(self.slider_wav.val)
		self.update_image()
		
		# Update checkbox state without triggering the callback logic
		self.internal_update = True
		is_good = self.good_frames[self.curr_sweep, self.curr_wav]
		current_status = self.chk_keep.get_status()[0]
		
		if is_good != current_status:
			self.chk_keep.set_active(0) # Toggle to match data
			
		self.internal_update = False
		self.update_stats()

	def on_keep_toggle(self, label):
		"""Handle checkbox click."""
		if self.internal_update:
			return
			
		# Toggle state in data model
		self.good_frames[self.curr_sweep, self.curr_wav] = not self.good_frames[self.curr_sweep, self.curr_wav]
		self.update_stats()

	def on_contrast_change(self, val):
		"""Handle vmin/vmax slider changes."""
		self.vmin = self.slider_vmin.val
		self.vmax = self.slider_vmax.val
		if self.vmin >= self.vmax: # Prevent error
			self.vmin = self.vmax - 0.1
		self.img_handle.set_clim(vmin=self.vmin, vmax=self.vmax)
		self.fig.canvas.draw_idle()

	def on_cmap_change(self, label):
		"""Handle colormap radio buttons."""
		mapping = {'Grayscale': 'gray', 'Magma': 'magma', 'Viridis': 'viridis'}
		self.cmap_name = mapping[label]
		self.img_handle.set_cmap(self.cmap_name)
		self.fig.canvas.draw_idle()

	def update_image(self):
		"""Updates the main image display."""
		frame = self.cube[self.curr_sweep, self.curr_wav, :, :]
		
		if self.img_handle is None:
			self.img_handle = self.ax_img.imshow(
				frame, 
				cmap=self.cmap_name, 
				vmin=self.vmin, 
				vmax=self.vmax, 
				origin=OriginPosition
			)
			self.ax_img.set_title(f"Sweep {self.curr_sweep} | {self.wav_labels[self.curr_wav]}")
		else:
			self.img_handle.set_data(frame)
			self.ax_img.set_title(f"Sweep {self.curr_sweep} | {self.wav_labels[self.curr_wav]}")
		
		self.fig.canvas.draw_idle()

	def update_stats(self):
		"""Updates the text statistics on the right."""
		stats_str = ""
		
		for w in range(self.n_wavs):
			count = np.sum(self.good_frames[:, w])
			label = self.wav_labels[w]
			
			# Highlight current selection
			prefix = ">> " if w == self.curr_wav else "   "
			
			# Append line
			stats_str += f"{prefix}{label}: {count}/{self.n_sweeps}\n"
			
		self.txt_stats.set_text(stats_str)
		self.fig.canvas.draw_idle()

	def finish(self, event):
		"""Closes GUI and saves results."""
		print("Selection Complete.")
		plt.close(self.fig)
		self.results = self._prepare_outputs()

	def _prepare_outputs(self):
		# 1. Good Indices List [N_good, 2] -> (sweep_idx, wav_idx)
		good_indices_list = np.argwhere(self.good_frames)
		
		return self.good_frames, good_indices_list

	def get_results(self):
		"""Retrieve results after closing."""
		if hasattr(self, 'results'):
			return self.results
		else:
			print("GUI not finished yet.")
			return None



