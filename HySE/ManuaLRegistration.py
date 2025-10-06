"""

Manual Registration (indicating fixed points manually)


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


# class LandmarkPicker:
#     """
#     An advanced interactive class to select corresponding points in two images
#     with undo, color-coding, and enforced point selection.
#     (CORRECTED to prevent WeakRef cleanup errors)
#     """
#     def __init__(self, fixed_image, moving_image=None, fixed_points_to_display=None, 
#                  frame_info=None, warning_message=None, deviation_threshold=150):
#         self.fixed_image = fixed_image
#         self.moving_image = moving_image
#         self.fixed_points = []
#         self.moving_points = []
#         self.plotted_artists = []
#         self.show_text_labels = False 
#         self.deviation_threshold = deviation_threshold
#         self.warning_messages = []

#         self.fig = plt.figure(figsize=(16, 8))
#         gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.1)
#         self.ax_fixed = self.fig.add_subplot(gs[0, 0])
#         self.ax_moving = self.fig.add_subplot(gs[0, 1])
#         self.axes = [self.ax_fixed, self.ax_moving]
#         self.fig.subplots_adjust(right=0.88)
#         self.cbar_ax = self.fig.add_axes([0.9, 0.15, 0.02, 0.7])

#         self.phase = 'moving' if moving_image is not None else 'fixed'

#         if self.phase == 'fixed':
#             self.num_total_points = None
#             self.fig.suptitle("PHASE 1: Select landmarks on the image.\nPress 'z' to undo, 'p' to toggle labels. MUST CLOSE window to finish.", fontsize=14)
#             self.ax_fixed.imshow(self._normalize(self.fixed_image), cmap='gray')
#             self.ax_fixed.set_title('Click to select FIXED points')
#             self.ax_moving.axis('off')
#         else: # moving phase
#             self.fixed_points_to_display = fixed_points_to_display
#             self.num_total_points = len(self.fixed_points_to_display)
#             frame_str = f"Frame {frame_info[0]} / {frame_info[1]}"
#             self.fig.suptitle(f"PHASE 2: {frame_str}\nPress 'z' to undo, 'p' to toggle labels. MUST select all {self.num_total_points} points.", fontsize=14)
#             self.ax_fixed.imshow(self._normalize(self.fixed_image), cmap='gray')
#             self.ax_fixed.set_title('FIXED points (reference)')
#             self.ax_moving.imshow(self._normalize(self.moving_image), cmap='gray')
		
#         if warning_message:
#             self.fig.text(0.5, 0.95, warning_message, color='red', ha='center', fontsize=12, weight='bold',
#                           bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

#         # Connect events and store their connection IDs (cids)
#         self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
#         self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
#         self.cid_close = self.fig.canvas.mpl_connect('close_event', self.on_close)
#         self._redraw()
		
	# def disconnect(self):
	#     """Disconnects all the matplotlib event connections."""
	#     self.fig.canvas.mpl_disconnect(self.cid_click)
	#     self.fig.canvas.mpl_disconnect(self.cid_key)
	#     self.fig.canvas.mpl_disconnect(self.cid_close)

	# def on_close(self, event):
	#     if self.phase == 'moving' and len(self.moving_points) < self.num_total_points:
	#         print(f"ACTION BLOCKED: Please select all {self.num_total_points} points before closing the window.")
	#     else:
	#         self.disconnect() # Disconnect before closing
	#         plt.close(self.fig)
			
	# def _normalize(self, img):
	#     p2, p98 = np.percentile(img, (2, 98)); return np.clip((img - p2) / (p98 - p2), 0, 1)
		
	# def on_click(self, event):
	#     if event.inaxes is None or event.button != 1: return
	#     x, y = event.xdata, event.ydata

	#     if self.phase == 'fixed' and event.inaxes == self.ax_fixed:
	#         self.fixed_points.append((x, y))

	#     elif self.phase == 'moving' and event.inaxes == self.ax_moving:
	#         if len(self.moving_points) < self.num_total_points:
	#             self.moving_points.append((x, y))
	#             ## Perform deviation test
	#             idx = len(self.moving_points) - 1
	#             p_fixed = self.fixed_points_to_display[idx]
	#             p_moving = self.moving_points[idx]
	#             dist = np.linalg.norm(np.array(p_fixed) - np.array(p_moving))

	#             if dist > self.deviation_threshold:
	#                 warning = f"Point #{idx + 1} deviation: {dist:.1f} px (>{self.deviation_threshold} px)"
	#                 self.warning_messages.append(warning)
	#                 print(f"   /!\\ WARNING: {warning}")
	#         else:
	#             print("All fixed points have a corresponding moving point. Cannot add more.")

	#     self._redraw()


	# def on_key(self, event):
	#     if event.key == 'z':
	#         if self.phase == 'fixed' and self.fixed_points:
	#             self.fixed_points.pop()
	#         elif self.phase == 'moving' and self.moving_points:
	#             # --- NEW: Remove warning associated with the undone point ---
	#             point_idx_to_remove = len(self.moving_points)
	#             # Filter out the warning for this specific point index
	#             self.warning_messages = [
	#                 msg for msg in self.warning_messages 
	#                 if not msg.startswith(f"Point #{point_idx_to_remove}")
	#             ]
	#             self.moving_points.pop()
	#         self._redraw()
	#     elif event.key == 'p':
	#         self.show_text_labels = not self.show_text_labels
	#         print(f"Text labels toggled {'ON' if self.show_text_labels else 'OFF'}")
	#         self._redraw()
	
	# def _redraw(self):
	#     for artist in self.plotted_artists: artist.remove()
	#     self.plotted_artists.clear()
	#     self.cbar_ax.clear()
	#     if self.phase == 'moving':
	#         self.ax_moving.set_title(f'Click to select MOVING point {len(self.moving_points)+1} of {self.num_total_points}')
	#         if len(self.moving_points) == self.num_total_points:
	#             self.ax_moving.set_title(f'All {self.num_total_points} points selected. You may now close the window.')

	#     if self.warning_messages:
	#         full_warning_text = "DEVIATION WARNINGS:\n" + "\n".join(self.warning_messages)
	#         warning_artist = self.fig.text(0.5, 0.05, full_warning_text, color='red', 
	#                                        ha='center', fontsize=10, weight='bold',
	#                                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
	#         self.plotted_artists.append(warning_artist)
		
	#     points_to_color = self.fixed_points if self.phase == 'fixed' else self.fixed_points_to_display
	#     num_colors = len(points_to_color) if len(points_to_color) > 1 else 2
	#     ## 'spring', 'autumn', 'cool', 'PiYG', 'vanimo', 'managua', 'jet'
	#     cmap = plt.colormaps['jet']
	#     colors = cmap(np.linspace(0, 1, num_colors))
	#     for i, (x, y) in enumerate(points_to_color):
	#         p = self.ax_fixed.plot(x, y, 'o', markersize=7, mfc=colors[i], mec='white', mew=0.5, alpha=0.6)
	#         self.plotted_artists.extend(p)
	#         if self.show_text_labels:
	#             t = self.ax_fixed.text(x + 5, y, str(i+1), color='white', fontsize=10, weight='bold')
	#             self.plotted_artists.append(t)
	#     for i, (x, y) in enumerate(self.moving_points):
	#         p = self.ax_moving.plot(x, y, 'o', markersize=7, mfc=colors[i], mec='white', mew=0.5, alpha=0.6)
	#         self.plotted_artists.extend(p)
	#         if self.show_text_labels:
	#             t = self.ax_moving.text(x + 5, y, str(i+1), color='white', fontsize=10, weight='bold')
	#             self.plotted_artists.append(t)
	#     if len(points_to_color) > 0:
	#         num_ticks = len(points_to_color)
	#         norm = mcolors.Normalize(vmin=1, vmax=num_ticks)
	#         sm = cm.ScalarMappable(cmap=cmap, norm=norm)
	#         sm.set_array([])
	#         ticks = np.linspace(1, num_ticks, num_ticks) if num_ticks > 1 else [1]
	#         cbar = self.fig.colorbar(sm, cax=self.cbar_ax, ticks=ticks)
	#         cbar.set_label('Point Index')
	#     self.fig.canvas.draw()
		
	# def get_points(self):
	#     plt.show(block=True)
	#     # --- MODIFIED: Disconnect handlers after the window is closed ---
	#     self.disconnect()
	#     return self.fixed_points if self.phase == 'fixed' else self.moving_points
	
	
class LandmarkPicker:
	"""
	An advanced interactive class to select corresponding points in two images
	with undo, color-coding, and enforced point selection.
	(CORRECTED to prevent WeakRef cleanup errors)
	"""
	def __init__(self, fixed_image, moving_image=None, fixed_points_to_display=None, 
				 frame_info=None, warning_message=None, deviation_threshold=150):
		self.fixed_image = fixed_image
		self.moving_image = moving_image
		self.fixed_points = []
		self.moving_points = []
		self.plotted_artists = []
		self.show_text_labels = False 
		self.deviation_threshold = deviation_threshold
		self.warning_messages = []

		self.fig = plt.figure(figsize=(16, 8))
		gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.1)
		self.ax_fixed = self.fig.add_subplot(gs[0, 0])
		self.ax_moving = self.fig.add_subplot(gs[0, 1])
		self.axes = [self.ax_fixed, self.ax_moving]
		self.fig.subplots_adjust(right=0.88)
		self.cbar_ax = self.fig.add_axes([0.9, 0.15, 0.02, 0.7])

		self.phase = 'moving' if moving_image is not None else 'fixed'

		if self.phase == 'fixed':
			self.num_total_points = None
			self.fig.suptitle("PHASE 1: Select landmarks on the image.\nPress 'z' to undo, 'p' to toggle labels. MUST CLOSE window to finish.", fontsize=14)
			self.ax_fixed.imshow(self._normalize(self.fixed_image), cmap='gray')
			self.ax_fixed.set_title('Click to select FIXED points')
			self.ax_moving.axis('off')
		else: # moving phase
			self.fixed_points_to_display = fixed_points_to_display
			self.num_total_points = len(self.fixed_points_to_display)
			frame_str = f"Frame {frame_info[0]} / {frame_info[1]}"
			self.fig.suptitle(f"PHASE 2: {frame_str}\nPress 'z' to undo, 'p' to toggle labels. MUST select all {self.num_total_points} points.", fontsize=14)
			self.ax_fixed.imshow(self._normalize(self.fixed_image), cmap='gray')
			self.ax_fixed.set_title('FIXED points (reference)')
			self.ax_moving.imshow(self._normalize(self.moving_image), cmap='gray')
		
		if warning_message:
			self.fig.text(0.5, 0.95, warning_message, color='red', ha='center', fontsize=12, weight='bold',
						  bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

		# Connect events and store their connection IDs (cids)
		self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
		self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
		self.cid_close = self.fig.canvas.mpl_connect('close_event', self.on_close)
		self._redraw()
		
	def disconnect(self):
		"""Disconnects all the matplotlib event connections."""
		self.fig.canvas.mpl_disconnect(self.cid_click)
		self.fig.canvas.mpl_disconnect(self.cid_key)
		self.fig.canvas.mpl_disconnect(self.cid_close)

	def on_close(self, event):
		if self.phase == 'moving' and len(self.moving_points) < self.num_total_points:
			print(f"ACTION BLOCKED: Please select all {self.num_total_points} points before closing the window.")
		else:
			self.disconnect() # Disconnect before closing
			plt.close(self.fig)
			
	def _normalize(self, img):
		p2, p98 = np.percentile(img, (2, 98)); return np.clip((img - p2) / (p98 - p2), 0, 1)
		
	def on_click(self, event):
		if event.inaxes is None or event.button != 1: return
		x, y = event.xdata, event.ydata

		if self.phase == 'fixed' and event.inaxes == self.ax_fixed:
			self.fixed_points.append((x, y))

		elif self.phase == 'moving' and event.inaxes == self.ax_moving:
			if len(self.moving_points) < self.num_total_points:
				self.moving_points.append((x, y))
				## Perform deviation test
				idx = len(self.moving_points) - 1
				p_fixed = self.fixed_points_to_display[idx]
				p_moving = self.moving_points[idx]
				# print(p_fixed)
				# print(p_moving)
				dist = np.linalg.norm(np.array(p_fixed) - np.array(p_moving))
				# print(dist)

				if dist > self.deviation_threshold:
					warning = f"Point #{idx + 1} deviation: {dist:.1f} px (>{self.deviation_threshold} px)"
					self.warning_messages.append(warning)
					print(f"   /!\\ WARNING: {warning}")
			else:
				print("All fixed points have a corresponding moving point. Cannot add more.")

		self._redraw()


	def on_key(self, event):
		if event.key == 'z':
			if self.phase == 'fixed' and self.fixed_points:
				self.fixed_points.pop()
			elif self.phase == 'moving' and self.moving_points:
				# --- NEW: Remove warning associated with the undone point ---
				point_idx_to_remove = len(self.moving_points)
				# Filter out the warning for this specific point index
				self.warning_messages = [
					msg for msg in self.warning_messages 
					if not msg.startswith(f"Point #{point_idx_to_remove}")
				]
				self.moving_points.pop()
			self._redraw()
		elif event.key == 'p':
			self.show_text_labels = not self.show_text_labels
			print(f"Text labels toggled {'ON' if self.show_text_labels else 'OFF'}")
			self._redraw()
	
	def _redraw(self):
		for artist in self.plotted_artists: artist.remove()
		self.plotted_artists.clear()
		self.cbar_ax.clear()
		if self.phase == 'moving':
			self.ax_moving.set_title(f'Click to select MOVING point {len(self.moving_points)+1} of {self.num_total_points}')
			if len(self.moving_points) == self.num_total_points:
				self.ax_moving.set_title(f'All {self.num_total_points} points selected. You may now close the window.')

		if self.warning_messages:
			full_warning_text = "DEVIATION WARNINGS:\n" + "\n".join(self.warning_messages)
			warning_artist = self.fig.text(0.5, 0.05, full_warning_text, color='red', 
										   ha='center', fontsize=10, weight='bold',
										   bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
			self.plotted_artists.append(warning_artist)
		
		points_to_color = self.fixed_points if self.phase == 'fixed' else self.fixed_points_to_display
		num_colors = len(points_to_color) if len(points_to_color) > 1 else 2
		## 'spring', 'autumn', 'cool', 'PiYG', 'vanimo', 'managua', 'jet'
		cmap = plt.colormaps['jet']
		colors = cmap(np.linspace(0, 1, num_colors))
		for i, (x, y) in enumerate(points_to_color):
			p = self.ax_fixed.plot(x, y, 'o', markersize=7, mfc=colors[i], mec='white', mew=0.5, alpha=0.6)
			self.plotted_artists.extend(p)
			if self.show_text_labels:
				t = self.ax_fixed.text(x + 5, y, str(i+1), color='white', fontsize=10, weight='bold')
				self.plotted_artists.append(t)
		for i, (x, y) in enumerate(self.moving_points):
			p = self.ax_moving.plot(x, y, 'o', markersize=7, mfc=colors[i], mec='white', mew=0.5, alpha=0.6)
			self.plotted_artists.extend(p)
			if self.show_text_labels:
				t = self.ax_moving.text(x + 5, y, str(i+1), color='white', fontsize=10, weight='bold')
				self.plotted_artists.append(t)
		if len(points_to_color) > 0:
			num_ticks = len(points_to_color)
			norm = mcolors.Normalize(vmin=1, vmax=num_ticks)
			sm = cm.ScalarMappable(cmap=cmap, norm=norm)
			sm.set_array([])
			ticks = np.linspace(1, num_ticks, num_ticks) if num_ticks > 1 else [1]
			cbar = self.fig.colorbar(sm, cax=self.cbar_ax, ticks=ticks)
			cbar.set_label('Point Index')
		self.fig.canvas.draw()
		
	def get_points(self):
		plt.show(block=True)
		# --- MODIFIED: Disconnect handlers after the window is closed ---
		self.disconnect()
		return self.fixed_points if self.phase == 'fixed' else self.moving_points
	
	
	
def _compute_landmark_transform(fixed_points, moving_points, transform_type='affine'):
	"""Takes lists of points and computes the SimpleITK transform."""
	min_points = {'rigid': 2, 'similarity': 2, 'affine': 3}.get(transform_type, 3)
	if len(fixed_points) < min_points or len(moving_points) < min_points:
		print(f"Warning: Not enough points for {transform_type} transform. Skipping.")
		return None
	fixed_flat = [p for point in fixed_points for p in point]
	moving_flat = [p for point in moving_points for p in point]
	if transform_type == 'affine': transform = _sitk.AffineTransform(2)
	elif transform_type == 'similarity': transform = _sitk.Similarity2DTransform()
	elif transform_type == 'rigid': transform = _sitk.Euler2DTransform()
	else: raise ValueError("Unsupported TransformType.")
	final_transform = _sitk.LandmarkBasedTransformInitializer(transform, fixed_flat, moving_flat)
	return final_transform
	
def _compute_landmark_transform(fixed_points, moving_points, transform_type='affine'):
	"""Takes lists of points and computes the SimpleITK transform."""
	min_points = {'rigid': 2, 'similarity': 2, 'affine': 3}.get(transform_type, 3)
	if len(fixed_points) < min_points or len(moving_points) < min_points:
		print(f"Warning: Not enough points for {transform_type} transform. Skipping.")
		return None
	fixed_flat = [p for point in fixed_points for p in point]
	moving_flat = [p for point in moving_points for p in point]
	if transform_type == 'affine': transform = _sitk.AffineTransform(2)
	elif transform_type == 'similarity': transform = _sitk.Similarity2DTransform()
	elif transform_type == 'rigid': transform = _sitk.Euler2DTransform()
	else: raise ValueError("Unsupported TransformType.")
	final_transform = _sitk.LandmarkBasedTransformInitializer(transform, fixed_flat, moving_flat)
	return final_transform


def _compute_landmark_transform(fixed_points, moving_points, transform_type='affine'):
	"""Takes lists of points and computes the SimpleITK transform."""
	min_points = {'rigid': 2, 'similarity': 2, 'affine': 3}.get(transform_type, 3)
	if len(fixed_points) < min_points or len(moving_points) < min_points: return None
	fixed_flat = [p for point in fixed_points for p in point]
	moving_flat = [p for point in moving_points for p in point]
	if transform_type == 'affine': transform = _sitk.AffineTransform(2)
	elif transform_type == 'similarity': transform = _sitk.Similarity2DTransform()
	elif transform_type == 'rigid': transform = _sitk.Euler2DTransform()
	else: raise ValueError("Unsupported TransformType.")
	return _sitk.LandmarkBasedTransformInitializer(transform, fixed_flat, moving_flat)

def CoRegisterImages_Manual(im_static, im_shifted, **kwargs):
	"""
	Co-registers a shifted image to a defined static image using SimpleITK (elastix).
	Can be initialized with a pre-existing transform for two-stage registration.
	"""
	InitialTransformFileName = kwargs.get('InitialTransformFileName')
	# This is a placeholder for your full CoRegisterImages_Manual function.
	# Please use the full version provided in the previous response.
	# For this example to run, we'll create a simplified version:
	print(f"--- Running Elastix Registration (B-Spline) ---")
	if InitialTransformFileName:
		print(f"  -> Initialized with transform from: {InitialTransformFileName}")
	
	elastixImageFilter = _sitk.ElastixImageFilter()
	elastixImageFilter.SetFixedImage(_sitk.GetImageFromArray(im_static))
	elastixImageFilter.SetMovingImage(_sitk.GetImageFromArray(im_shifted))
	
	parameterMap = _sitk.GetDefaultParameterMap('bspline')
	parameterMap['FinalGridSpacingInPhysicalUnits'] = [str(kwargs.get('GridSpacing', 20))]
	if InitialTransformFileName:
		parameterMap['InitialTransformParametersFileName'] = [InitialTransformFileName]

	elastixImageFilter.SetParameterMap(parameterMap)
	elastixImageFilter.LogToConsoleOff()
	elastixImageFilter.Execute()
	
	transformParameterMap = elastixImageFilter.GetTransformParameterMap()
	
	transformixImageFilter = _sitk.TransformixImageFilter()
	transformixImageFilter.LogToConsoleOff()
	transformixImageFilter.SetTransformParameterMap(transformParameterMap)
	transformixImageFilter.SetMovingImage(_sitk.GetImageFromArray(im_shifted.astype(np.float32)))
	result_orig_se = transformixImageFilter.Execute()
	
	return _sitk.GetArrayFromImage(result_orig_se), transformParameterMap


def CoRegisterHypercubeAndMask_Hybrid(RawHypercube, Wavelengths_list, **kwargs):
	"""
	Apply a hybrid landmark-initialized B-spline co-registration to a hypercube.

	This function uses a dedicated two-stage process:
	1.  The user interactively defines landmarks for a robust global (Affine) alignment.
	2.  This alignment is used to initialize a flexible, pixel-based B-Spline
		registration for fine-tuning local distortions.
	
	The interactive process can be bypassed for reproducibility by providing pre-selected 
	points via the `AllLandmarkPoints` kwarg.

	Input:
		- RawHypercube: To co-registrate. Shape [N, Y, X]
		- Wavelengths_list
		- kwargs:
			- Help, Cropping, Order, SaveHypercube, PlotDiff, SavingPath, EdgeMask, 
			  AllReflectionsMasks, HideReflections
			- Static_Index (0): Which image is set as the static one.
			- AllLandmarkPoints (None): A dictionary {'fixed_points': [], 'moving_points': [[]]} 
									   to bypass the interactive GUI.
			- TransformType ('Affine'): Global transform for the landmark stage.
			- deviation_threshold (50): Pixel distance threshold for landmark warnings.
			- Plus all kwargs for the B-Spline stage (e.g., GridSpacing, Metric).

	Outputs:
		- Hypercube_sorted: The final co-registered hypercube.
		- Coregistration_Transforms: The list of final B-Spline transformation maps.
		- CombinedMask: A single 2D mask where True indicates an invalid pixel.
		- AllLandmarkPoints: Dictionary of all selected landmark coordinates.
	"""
	if kwargs.get('Help', False): print(inspect.getdoc(CoRegisterHypercubeAndMask_Hybrid)); return (None,)*4

	
	Static_Index = kwargs.get('StaticIndex', 0)
	Cropping = kwargs.get('Cropping', 0)
	PostCropping = kwargs.get('PostCropping')
	Blurring = kwargs.get('Blurring', True)
	Sigma = kwargs.get('Sigma', 2)
	Order = kwargs.get('Order', False)
	EdgeMask = kwargs.get('EdgeMask')
	AllReflectionsMasks = kwargs.get('AllReflectionsMasks')
	HideReflections = kwargs.get('HideReflections', True)
	deviation_threshold = kwargs.get('DeviationThreshold', 200)
	

	
	t0 = time.time()
	(NN, YY, XX) = RawHypercube.shape
	
	if Cropping != 0:
		print(f'Image will be cropped by {Cropping} on all sides.')
		RawHypercube = RawHypercube[:, Cropping:-Cropping, Cropping:-Cropping]
		if AllReflectionsMasks is not None: AllReflectionsMasks = AllReflectionsMasks[:, Cropping:-Cropping, Cropping:-Cropping]

	(NN_crop, YY_crop, XX_crop) = RawHypercube.shape
	CombinedMask = np.zeros((YY_crop, XX_crop), dtype=bool)
	im_static = RawHypercube[Static_Index, :, :]

	Hypercube = []
	AllTransforms = []
	
	AllLandmarkPoints_input = kwargs.get('AllLandmarkPoints')
	PromptUser = AllLandmarkPoints_input is None
	fixed_landmarks = AllLandmarkPoints_input['fixed_points'] if not PromptUser else []
	all_moving_landmarks_input = AllLandmarkPoints_input['moving_points'] if not PromptUser else []
	all_moving_landmarks_output = []

	if PromptUser:
		picker = LandmarkPicker(im_static)
		fixed_landmarks = picker.get_points()
		min_points = 3 # For the initial Affine transform
		if len(fixed_landmarks) < min_points:
			raise ValueError(f"Not enough fixed points selected ({len(fixed_landmarks)}). At least {min_points} are required. Aborting.")
	
	warning_for_next_frame = None

	for c in range(0, NN):
		if c == Static_Index:
			im_static_processed = copy.deepcopy(im_static)
			if Blurring:
				im_static_processed = gaussian_filter(im_static_processed, sigma=Sigma)
			if HideReflections and AllReflectionsMasks is not None:
				ReflectionsMask_Static = AllReflectionsMasks[Static_Index, :, :]
				if ReflectionsMask_Static is not None: im_static_processed[ReflectionsMask_Static > 0.5] = np.nan
			Hypercube.append(im_static_processed)
			AllTransforms.append(0)
			all_moving_landmarks_output.append(fixed_landmarks)
			continue

		print(f'\nWorking on Frame: {c+1} / {NN}')
		im_shifted = RawHypercube[c, :, :]
		
		# --- STAGE 1: LANDMARKS ---
		if PromptUser:
			picker = LandmarkPicker(im_static, im_shifted, 
									fixed_points_to_display=fixed_landmarks,
									frame_info=(c + 1, NN),
									warning_message=warning_for_next_frame)
			moving_landmarks_for_frame = picker.get_points()
		else:
			moving_landmarks_for_frame = all_moving_landmarks_input[c]
		
		all_moving_landmarks_output.append(moving_landmarks_for_frame)
		warning_for_next_frame = None

		if len(moving_landmarks_for_frame) == len(fixed_landmarks) and len(fixed_landmarks) > 0:
			deviations = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(fixed_landmarks, moving_landmarks_for_frame)]
			if deviations:
				max_dev = np.max(deviations)
				if max_dev > deviation_threshold:
					warning_for_next_frame = (f"WARNING on PREVIOUS frame ({c+1}):\n"
											  f"Max landmark deviation was {max_dev:.1f} pixels (Threshold: {deviation_threshold})")
					print(f"\n/!\\ {warning_for_next_frame}\n")
		
		initial_transform = _compute_landmark_transform(fixed_landmarks, moving_landmarks_for_frame, kwargs.get('TransformType', 'Affine').lower())
		
		# --- STAGE 2: B-SPLINE REFINEMENT ---
		if initial_transform:
			with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w+') as temp_file:
				initial_transform_filename = temp_file.name
				_sitk.WriteTransform(initial_transform, initial_transform_filename)
			
			im_coregistered_smeared, final_transform_map = CoRegisterImages_Manual(
				im_static, im_shifted,
				InitialTransformFileName=initial_transform_filename,
				**kwargs
			)
			im_coregistered_final = im_coregistered_smeared.astype(np.float32)
			os.remove(initial_transform_filename)
			transform_to_store = final_transform_map
		else:
			print("  -> Landmark step failed, skipping Elastix refinement. Image will not be registered.")
			im_coregistered_final = im_shifted
			transform_to_store = None
		
		# --- Post-Registration Processing ---
		if HideReflections and AllReflectionsMasks is not None and transform_to_store is not None:
			ReflectionsMask_Shifted = AllReflectionsMasks[c, :, :]
			if ReflectionsMask_Shifted is not None:         
				
				## a. Create "Hole-Punch Mask". 
				##    This mask is True for every pixel to remove
				##    Start with a mask that is True everywhere outside the ROI.
				hole_punch_mask_moving = np.ones_like(im_shifted, dtype=np.uint8)
				y_start, y_end = 0, -1
				x_start, x_end = 0, -1
				hole_punch_mask_moving[y_start:y_end, x_start:x_end] = 0 # Set the ROI interior to False (we want to keep it)

				## b. Now, add the reflections to the mask. 
				##    Set reflection pixels to True (we want to remove them).
				if ReflectionsMask_Shifted is not None:
					hole_punch_mask_moving[ReflectionsMask_Shifted > 0.5] = 1

				## c. Now warp this combined hole-punch mask.
				transformixImageFilter = _sitk.TransformixImageFilter()
				transformixImageFilter.LogToConsoleOff()

				## Set the transform map from the image registration
				transformixImageFilter.SetTransformParameterMap(final_transform_map)

				## Modify the parameter map to use nearest neighbor interpolation
				final_transform_map[0]['FinalBSplineInterpolationOrder'] = ['0']
				if len(final_transform_map) > 1: # Handle TwoStage registration
					final_transform_map[1]['FinalBSplineInterpolationOrder'] = ['0']

				transformixImageFilter.SetTransformParameterMap(final_transform_map)

				moving_mask_sitk = _sitk.GetImageFromArray(hole_punch_mask_moving)

				transformixImageFilter.SetMovingImage(moving_mask_sitk)

				registered_mask_sitk = transformixImageFilter.Execute()
				registered_mask = _sitk.GetArrayFromImage(registered_mask_sitk).astype(bool)

				## e. Punch holes in the final registered image
				im_coregistered_final[registered_mask] = np.nan
				
			  
		if Blurring:
				im_coregistered_final = gaussian_filter(im_coregistered_final, sigma=Sigma)
		Hypercube.append(im_coregistered_final)
		AllTransforms.append(transform_to_store)
		
		invalid_pixels_mask = np.isnan(im_coregistered_final) | (im_coregistered_final == 0)
		CombinedMask = CombinedMask | invalid_pixels_mask
			
	tf = time.time(); time_total = tf - t0
	minutes = int(time_total / 60); seconds = time_total - minutes * 60
	print(f'\n\nCo-registration took {minutes} min and {seconds:.0f} s in total\n')

	AllLandmarkPoints_output = {'fixed_points': fixed_landmarks, 'moving_points': all_moving_landmarks_output}
	
	Order = kwargs.get('Order', False)
	order_list = np.argsort(Wavelengths_list) if Order else np.arange(len(Hypercube))
	Hypercube_sorted = np.array(Hypercube)[order_list]
	AllTransforms_sorted = [AllTransforms[i] for i in order_list]
	
	if kwargs.get('SaveHypercube', False):
		# Your saving logic here
		pass

	return Hypercube_sorted, AllTransforms_sorted, CombinedMask, AllLandmarkPoints_output

