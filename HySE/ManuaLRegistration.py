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
import matplotlib.pyplot as plt
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
import pickle
import re


import SimpleITK as _sitk
# import matplotlib.colors as mcolors
# import matplotlib.cm as cm


from matplotlib.widgets import Slider, RadioButtons

OriginPosition = 'lower'
# OriginPosition = 'upper' ## standard python

## N.A. Functions and GUI written with Gemini

class LandmarkPicker:
	"""
	An advanced interactive class to select corresponding points in two images
	with undo, color-coding, enforced point selection, and image controls.
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

		self.fig = plt.figure(figsize=(16, 9))

		# --- ALLOW ZOOM/PAN ---
		# We ensure the toolbar is active. The 'on_click' function checks its state.
		
		# Adjust layout to make room for widgets at the bottom
		gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.1)
		self.fig.subplots_adjust(bottom=0.25, right=0.88)
		
		self.ax_fixed = self.fig.add_subplot(gs[0, 0])
		self.ax_moving = self.fig.add_subplot(gs[0, 1])
		self.axes = [self.ax_fixed, self.ax_moving]
		self.cbar_ax = self.fig.add_axes([0.9, 0.25, 0.02, 0.6])

		self.phase = 'moving' if moving_image is not None else 'fixed'

		# --- DISPLAY IMAGES & STORE REFERENCES ---
		# We store the image objects (im_obj) to update clims/cmap later without clearing axes
		if self.phase == 'fixed':
			self.num_total_points = None
			self.fig.suptitle("PHASE 1: Select landmarks. Use toolbar to Zoom/Pan.\n'z': undo, 'p': labels. CLOSE to finish.", fontsize=14)
			self.fixed_img_norm = self._normalize(self.fixed_image)
			self.im_fixed_obj = self.ax_fixed.imshow(self.fixed_img_norm, cmap='magma', vmin=0, vmax=1, origin=OriginPosition)
			self.ax_fixed.set_title('Click to select FIXED points')
			self.ax_moving.axis('off')
			self.im_moving_obj = None
		else: # moving phase
			self.fixed_points_to_display = fixed_points_to_display
			self.num_total_points = len(self.fixed_points_to_display)
			frame_str = f"Frame {frame_info[0]} / {frame_info[1]}"
			self.fig.suptitle(f"PHASE 2: {frame_str}\nUse toolbar to Zoom/Pan. 'z': undo, 'p': labels. MUST select {self.num_total_points} points.", fontsize=14)
			
			self.fixed_img_norm = self._normalize(self.fixed_image)
			self.im_fixed_obj = self.ax_fixed.imshow(self.fixed_img_norm, cmap='magma', vmin=0, vmax=1, origin=OriginPosition)
			self.ax_fixed.set_title('FIXED points (reference)')
			
			self.moving_img_norm = self._normalize(self.moving_image)
			self.im_moving_obj = self.ax_moving.imshow(self.moving_img_norm, cmap='magma', vmin=0, vmax=1, origin=OriginPosition)
		
		if warning_message:
			self.fig.text(0.5, 0.95, warning_message, color='red', ha='center', fontsize=12, weight='bold',
						  bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

		# --- WIDGETS ---
		# Define positions [left, bottom, width, height]
		ax_color = 'lightgoldenrodyellow'
		
		# 1. Sliders for Fixed Image (Left side)
		ax_fix_min = self.fig.add_axes([0.15, 0.1, 0.25, 0.03], facecolor=ax_color)
		ax_fix_max = self.fig.add_axes([0.15, 0.06, 0.25, 0.03], facecolor=ax_color)
		self.slider_fix_min = Slider(ax_fix_min, 'Fixed Min', 0.0, 1.0, valinit=0.0)
		self.slider_fix_max = Slider(ax_fix_max, 'Fixed Max', 0.0, 1.0, valinit=1.0)
		
		self.slider_fix_min.on_changed(self.update_fixed_clim)
		self.slider_fix_max.on_changed(self.update_fixed_clim)

		# 2. Sliders for Moving Image (Right side) - Only if in moving phase
		if self.phase == 'moving':
			ax_mov_min = self.fig.add_axes([0.55, 0.1, 0.25, 0.03], facecolor=ax_color)
			ax_mov_max = self.fig.add_axes([0.55, 0.06, 0.25, 0.03], facecolor=ax_color)
			self.slider_mov_min = Slider(ax_mov_min, 'Moving Min', 0.0, 1.0, valinit=0.0)
			self.slider_mov_max = Slider(ax_mov_max, 'Moving Max', 0.0, 1.0, valinit=1.0)
			
			self.slider_mov_min.on_changed(self.update_moving_clim)
			self.slider_mov_max.on_changed(self.update_moving_clim)

		# 3. Radio Buttons for Colormap
		ax_radio = self.fig.add_axes([0.92, 0.05, 0.07, 0.15], facecolor=ax_color)
		self.radio = RadioButtons(ax_radio, ('gray', 'magma', 'viridis'))
		self.radio.on_clicked(self.update_cmap)

		self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
		self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
		self.cid_close = self.fig.canvas.mpl_connect('close_event', self.on_close)
		self._redraw()

	def update_fixed_clim(self, val):
		"""Callback to update contrast for the fixed image."""
		if self.im_fixed_obj:
			vmin = self.slider_fix_min.val
			vmax = self.slider_fix_max.val
			# Ensure vmin < vmax to avoid errors
			if vmin >= vmax: 
				vmax = vmin + 0.01
			self.im_fixed_obj.set_clim(vmin, vmax)
			self.fig.canvas.draw_idle()

	def update_moving_clim(self, val):
		"""Callback to update contrast for the moving image."""
		if self.im_moving_obj:
			vmin = self.slider_mov_min.val
			vmax = self.slider_mov_max.val
			if vmin >= vmax: 
				vmax = vmin + 0.01
			self.im_moving_obj.set_clim(vmin, vmax)
			self.fig.canvas.draw_idle()

	def update_cmap(self, label):
		"""Callback to update the colormap for both images."""
		if self.im_fixed_obj:
			self.im_fixed_obj.set_cmap(label)
		if self.im_moving_obj:
			self.im_moving_obj.set_cmap(label)
		self.fig.canvas.draw_idle()

	def on_click(self, event):
		if event.inaxes is None or event.button != 1: return
		
		# --- CHECK TOOLBAR & WIDGET STATE ---
		# 1. If zoom/pan is active, ignore click
		toolbar = self.fig.canvas.manager.toolbar
		if toolbar is not None and toolbar.mode != '':
			return
			
		# 2. If click is inside the main axes (not on sliders), proceed
		# This prevents clicks on sliders from adding points
		if event.inaxes not in [self.ax_fixed, self.ax_moving]:
			return

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
				dist = np.linalg.norm(np.array(p_fixed) - np.array(p_moving))

				if dist > self.deviation_threshold:
					warning = f"Point #{idx + 1} deviation: {dist:.1f} px (>{self.deviation_threshold} px)"
					self.warning_messages.append(warning)
					print(f"    /!\\ WARNING: {warning}")
			else:
				print("All fixed points have a corresponding moving point. Cannot add more.")

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
		
		# --- CHECK TOOLBAR STATE ---
		# If the toolbar exists and is in a mode (zoom/pan), don't add a point.
		toolbar = self.fig.canvas.manager.toolbar
		if toolbar is not None and toolbar.mode != '':
			return

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
				dist = np.linalg.norm(np.array(p_fixed) - np.array(p_moving))

				if dist > self.deviation_threshold:
					warning = f"Point #{idx + 1} deviation: {dist:.1f} px (>{self.deviation_threshold} px)"
					self.warning_messages.append(warning)
					print(f"    /!\\ WARNING: {warning}")
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

		if event.key == 'w':
			print(f'Erasing Warning Messages')
			self.warning_messages = ''
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



from scipy.interpolate import Rbf #RBFInterpolator #

def _compute_tps_transform(fixed_points, moving_points, reference_image_array):
	"""
	Computes a Thin Plate Spline (TPS) displacement field transform that 
	perfectly aligns the fixed points to the moving points.
	
	Args:
		fixed_points: List of tuples (x, y) in physical coordinates.
		moving_points: List of tuples (x, y) in physical coordinates.
		reference_image: The sitk.Image defining the domain (size, spacing, origin).
		
	Returns:
		sitk.DisplacementFieldTransform: A non-rigid transform aligning landmarks exactly.
	"""
	if len(fixed_points) < 3 or len(moving_points) < 3:
		raise ValueError("TPS requires at least 3 points.")

	# Convert imput image to sikt image:
	reference_image = _sitk.GetImageFromArray(reference_image_array)

	# 1. Separate points into arrays
	# Fixed points (Target for the RBF query)
	fx = np.array([p[0] for p in fixed_points])
	fy = np.array([p[1] for p in fixed_points])
	
	# Moving points (Values at the target)
	mx = np.array([p[0] for p in moving_points])
	my = np.array([p[1] for p in moving_points])

	# 2. Calculate displacements (Moving - Fixed)
	# The transform needs to know: "For a point at Fixed(x,y), where is it in Moving?"
	# The displacement vector is D = Moving - Fixed
	dx = mx - fx
	dy = my - fy

	# 3. Fit Radial Basis Functions (Thin Plate Spline)
	# This creates a function that interpolates the displacements exactly
	rbf_x = Rbf(fx, fy, dx, function='thin_plate')
	rbf_y = Rbf(fx, fy, dy, function='thin_plate')
	# rbf_x = RBFInterpolator(fx, fy, dx, kernel='thin_plate_spline')
	# rbf_y = RBFInterpolator(fx, fy, dy, kernel='thin_plate_spline')

	# 4. Generate a grid of physical coordinates from the reference image
	size = reference_image.GetSize()
	spacing = reference_image.GetSpacing()
	origin = reference_image.GetOrigin()
	direction = np.array(reference_image.GetDirection()).reshape(2, 2)

	# Create index grid
	x_idx = np.arange(0, size[0])
	y_idx = np.arange(0, size[1])
	xv_idx, yv_idx = np.meshgrid(x_idx, y_idx, indexing='xy')

	# Convert index grid to physical coordinates (handling rotation/direction)
	# Physical = Origin + Direction * (Index * Spacing)
	# Note: This matrix mult is manual to support rotated images
	phys_x = origin[0] + direction[0, 0] * xv_idx * spacing[0] + direction[0, 1] * yv_idx * spacing[1]
	phys_y = origin[1] + direction[1, 0] * xv_idx * spacing[0] + direction[1, 1] * yv_idx * spacing[1]

	# 5. Evaluate RBF on the grid to get the displacement field
	disp_x = rbf_x(phys_x, phys_y)
	disp_y = rbf_y(phys_x, phys_y)

	# 6. Create SimpleITK Displacement Field
	# Stack x and y displacements into a vector image
	# Note: SimpleITK expects the vector image to be (SizeX, SizeY) with vector pixels
	# We must transpose numpy arrays because SimpleITK is (x, y) but numpy is (row, col) aka (y, x)
	disp_np = np.stack((disp_x, disp_y), axis=-1)
	
	# Create the image from the array
	displacement_img = _sitk.GetImageFromArray(disp_np, isVector=True)
	displacement_img.CopyInformation(reference_image)

	# 7. Create the Transform
	transform = _sitk.DisplacementFieldTransform(displacement_img)
	
	return transform


	
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



def ManualRegistration(RawHypercube, Wavelengths_list, **kwargs):
	"""
	Apply a LANDMARK-ONLY co-registration to a hypercube.

	This function performs a single-stage registration based *only* on the
	transform computed from user-defined landmarks (e.g., Affine).
	
	It intentionally SKIPS the B-Spline refinement step to ensure the
	final warp is determined exclusively by the provided landmarks.

	Input:
		- RawHypercube: To co-registrate. Shape [N, Y, X]
		- Wavelengths_list
		- kwargs:
			- Help, Cropping, Order, SaveHypercube, PlotDiff, SavingPath, EdgeMask, 
			  AllReflectionsMasks, HideReflections
			- Static_Index (0): Which image is set as the static one.
			- Exact = True: If set to False, the code will use the input points to approximate the 
					best affine transform. If set to True, it will compute a Thin Plate Spline (TPS)
					transform to interpolate the best transform that goes exactly through the points.
			- AllLandmarkPoints (None): A dictionary {'fixed_points': [], 'moving_points': [[]]} 
									  to bypass the interactive GUI.
			- TransformType ('Affine'): Global transform for the landmark stage.
			- deviation_threshold (50): Pixel distance threshold for landmark warnings.
			- ** B-Spline kwargs (e.g., GridSpacing) will be IGNORED. **

	Outputs:
		- Hypercube_sorted: The final co-registered hypercube.
		- Coregistration_Transforms: The list of final landmark-based
		  SimpleITK transform *objects*.
		- CombinedMask: A single 2D mask where True indicates an invalid pixel.
		- AllLandmarkPoints: Dictionary of all selected landmark coordinates.
	"""
	if kwargs.get('Help', False): print(inspect.getdoc(CoRegisterHypercube_LandmarkOnly)); return (None,)*4

	
	Static_Index = kwargs.get('StaticIndex', 0)
	print(f'Setting frame  {Static_Index} as static frame')
	Cropping = kwargs.get('Cropping', 0)
	Exact = kwargs.get('Exact', True)
	Blurring = kwargs.get('Blurring', True)
	Sigma = kwargs.get('Sigma', 2)
	Order = kwargs.get('Order', False)
	EdgeMask = kwargs.get('EdgeMask') # Note: EdgeMask is not used in this version
	AllReflectionsMasks = kwargs.get('AllReflectionsMasks')
	HideReflections = kwargs.get('HideReflections', True)
	deviation_threshold = kwargs.get('DeviationThreshold', 200)
	if Exact:
		print('Computing a Thin Plate Spline transform that exactly goes through the identified points')
	else:
		print('Computing an Affine transform that best approximates going through all the identified points')
	
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
		# Assuming LandmarkPicker class is defined elsewhere
		picker = LandmarkPicker(im_static)
		fixed_landmarks = picker.get_points()
		min_points = 3 # For the initial Affine transform
		if len(fixed_landmarks) < min_points:
			raise ValueError(f"Not enough fixed points selected ({len(fixed_landmarks)}). At least {min_points} are required. Aborting.")
	
	warning_for_next_frame = None
	
	# --- Create a reference SimpleITK static image ---
	# This is used as the output grid for all warped images
	sitk_im_static = _sitk.GetImageFromArray(im_static.astype(np.float32))

	for c in range(0, NN):
		if c == Static_Index:
			# --- Process Static Frame ---
			im_static_processed = copy.deepcopy(im_static).astype(np.float32) # Ensure float
			
			if HideReflections and AllReflectionsMasks is not None:
				ReflectionsMask_Static = AllReflectionsMasks[Static_Index, :, :]
				if ReflectionsMask_Static is not None: 
					im_static_processed[ReflectionsMask_Static > 0.5] = np.nan
			
			if Blurring:
				# Apply NaN-aware blur
				try:
					im_static_processed = HySE.CoRegistrationTools.gaussian_blur_nan(im_static_processed, sigma=Sigma)
				except NameError:
					print("Static Frame Warning: 'gaussian_blur_nan' not found. Blurring may be incorrect.")
					im_static_processed = _sitk.GetArrayFromImage(_sitk.DiscreteGaussian(_sitk.GetImageFromArray(im_static_processed), Sigma))

			Hypercube.append(im_static_processed)
			AllTransforms.append(0) # No transform for static image
			all_moving_landmarks_output.append(fixed_landmarks)
			
			# Update combined mask for the static frame
			invalid_pixels_mask = np.isnan(im_static_processed) | (im_static_processed == 0)
			CombinedMask = CombinedMask | invalid_pixels_mask
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
				# if max_dev > deviation_threshold:
					# warning_for_next_frame = (f"WARNING on PREVIOUS frame ({c+1}):\n"
											  # f"Max landmark deviation was {max_dev:.1f} pixels (Threshold: {deviation_threshold})")
					# print(f"\n/!\\ {warning_for_next_frame}\n")
		
		# Assuming _compute_landmark_transform returns a SimpleITK Transform object
		if Exact:
			initial_transform = _compute_tps_transform(fixed_landmarks, moving_landmarks_for_frame, im_shifted)
		else:
			initial_transform = _compute_landmark_transform(fixed_landmarks, moving_landmarks_for_frame, kwargs.get('TransformType', 'Affine').lower())
		
		
		# --- STAGE 2: APPLY LANDMARK TRANSFORM (B-SPLINE SKIPPED) ---
		if initial_transform:
			print("  -> Applying landmark-based transform (skipping B-Spline).")
			
			sitk_im_shifted = _sitk.GetImageFromArray(im_shifted.astype(np.float32))
			sitk_im_shifted.CopyInformation(sitk_im_static) # Match metadata
			
			# Resample the image using the landmark transform
			# Use Linear interpolation for the image data
			sitk_im_coregistered = _sitk.Resample(sitk_im_shifted, sitk_im_static, initial_transform, 
												 _sitk.sitkLinear, 0.0)
			
			im_coregistered_final = _sitk.GetArrayFromImage(sitk_im_coregistered)
			transform_to_store = initial_transform
		else:
			print("  -> Landmark step failed, skipping transform. Image will not be registered.")
			im_coregistered_final = im_shifted.astype(np.float32)
			transform_to_store = None
		
		
		# --- Post-Registration Processing ---
		if HideReflections and AllReflectionsMasks is not None and transform_to_store is not None:
			ReflectionsMask_Shifted = AllReflectionsMasks[c, :, :]
			if ReflectionsMask_Shifted is not None: 
				print("  -> Warping reflection mask...")
				
				# Create the "hole punch" mask (1s = remove, 0s = keep)
				hole_punch_mask_moving = np.zeros_like(im_shifted, dtype=np.uint8)
				if ReflectionsMask_Shifted is not None:
					hole_punch_mask_moving[ReflectionsMask_Shifted > 0.5] = 1

				# --- Warp the mask using SimpleITK Resample ---
				sitk_mask_moving = _sitk.GetImageFromArray(hole_punch_mask_moving)
				sitk_mask_moving.CopyInformation(sitk_im_static)
				
				# CRITICAL: Use Nearest Neighbor interpolation for masks
				registered_mask_sitk = _sitk.Resample(sitk_mask_moving, sitk_im_static, transform_to_store,
													 _sitk.sitkNearestNeighbor, 0.0)
				
				registered_mask = _sitk.GetArrayFromImage(registered_mask_sitk).astype(bool)

				## e. Punch holes in the final registered image
				im_coregistered_final[registered_mask] = np.nan
				
			
		if Blurring:
			try:
				im_coregistered_final = HySE.CoRegistrationTools.gaussian_blur_nan(im_coregistered_final, sigma=Sigma)
			except NameError:
				print("Moving Frame Warning: 'gaussian_blur_nan' not found. Blurring may be incorrect.")
				im_coregistered_final = _sitk.GetArrayFromImage(_sitk.DiscreteGaussian(_sitk.GetImageFromArray(im_coregistered_final), Sigma))

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
		print("Saving hypercube (logic not implemented in this snippet)...")
		pass

	return Hypercube_sorted, AllTransforms_sorted, CombinedMask, AllLandmarkPoints_output






def MakeCombinedHypercubeForRegistration(hypercube, selector_results, original_wavelengths=None):
	"""
	Flattens a 4D Hypercube into a 3D stack based on valid frames selected in the GUI.
	
	Parameters:
	-----------
	hypercube : np.ndarray
		Input 4D array of shape [Nsweeps, Nwavelengths, Y, X].
	selector_results : tuple
		The output from selector.get_results(), containing (mask, indices).
	original_wavelengths : list, optional
		List of strings describing the wavelengths (length Nwavelengths).
		Used to generate unique labels for the flattened stack (e.g., "Sweep0_Wav2").
		
	Returns:
	--------
	filtered_stack : np.ndarray
		3D array [N_valid, Y, X] ready for ManualRegistration.
	new_labels : list
		List of strings describing each frame in the filtered stack, 
		matching the length of filtered_stack.
	"""
	# Unpack the results from the GUI
	_, good_indices = selector_results
	
	# good_indices is shape [N_valid, 2] where cols are [sweep_idx, wav_idx]
	# We use advanced indexing to extract all valid frames at once
	# Resulting shape: [N_valid, Y, X]
	filtered_stack = hypercube[good_indices[:, 0], good_indices[:, 1]]
	
	# Generate updated Wavelengths_list for ManualRegistration
	new_labels = []
	
	# Default labels if none provided
	if original_wavelengths is None:
		num_wavs = hypercube.shape[1]
		original_wavelengths = [f"Frame{i}" for i in range(num_wavs)]
		
	for i in range(len(good_indices)):
		s_idx, w_idx = good_indices[i]
		# Create a label that preserves the sweep and wavelength identity
		# e.g., "S0_Frame0"
		label = f"S{s_idx}_{original_wavelengths[w_idx]}"
		new_labels.append(label)
		
	return filtered_stack, new_labels





def SaveAllTransforms(transforms_list, labels_list, filename="RegistrationTransforms.pkl"):
    """
    Saves a list of SimpleITK transforms and their associated labels to a single file.

    Parameters:
    -----------
    transforms_list : list
        List of SimpleITK transform objects (output from ManualRegistration).
    labels_list : list
        List of strings identifying each transform (output from MakeCombinedHypercubeForRegistration).
    filename : str
        Path to save the output file.
    """
    if len(transforms_list) != len(labels_list):
        raise ValueError("Error: The number of transforms must match the number of labels.")

    # Create a dictionary mapping Label -> Transform
    # This ensures we always know exactly which transform belongs to which frame
    transform_dict = dict(zip(labels_list, transforms_list))

    # Save to a single pickle file
    with open(filename, 'wb') as f:
        pickle.dump(transform_dict, f)
    
    print(f"Successfully saved {len(transforms_list)} transforms to {filename}")




def ApplyAllTransforms(data_hypercube, selector_results, transforms_file_path, original_wavelengths=None):
    """
    Applies loaded transforms to the valid frames of a Data Hypercube.

    Parameters:
    -----------
    data_hypercube : np.ndarray
        4D array [Nsweeps, Nwavelengths, Y, X] containing the data to be warped.
    selector_results_indices : lists
        The indices output from the (mask, indices) tuple from the FrameSelector GUI.
    transforms_file_path : str
        Path to the .pkl file created by SaveAllTransforms.
    original_wavelengths : list, optional
        List of original frame names (e.g., ['Frame0', 'Frame1'...]).
        Must match what was used during registration to generate the keys.

    Returns:
    --------
    transformed_stack : np.ndarray
        3D array [N_valid, Y, X] of coregistered data frames.
    valid_labels : list
        List of labels corresponding to the output stack.
    """
    # 1. Load Transforms
    with open(transforms_file_path, 'rb') as f:
        transform_dict = pickle.load(f)

    # 2. Unpack indices
    good_indices = selector_results
    n_valid = len(good_indices)
    _, _, h, w = data_hypercube.shape

    # 3. Handle default labels
    if original_wavelengths is None:
        num_wavs = data_hypercube.shape[1]
        original_wavelengths = [f"Frame{i}" for i in range(num_wavs)]

    # 4. Prepare Output Array
    transformed_stack = np.zeros((n_valid, h, w), dtype=data_hypercube.dtype)
    valid_labels = []

    print(f"Applying transforms to {n_valid} frames...")

    for i, (s_idx, w_idx) in enumerate(good_indices):
        # Extract the frame
        image_np = data_hypercube[s_idx, w_idx, :, :]
        
        # Reconstruct the label key to find the correct transform
        # e.g. "S2_Frame5"
        label_key = f"S{s_idx}_{original_wavelengths[w_idx]}"
        
        if label_key not in transform_dict:
            print(f"Warning: No transform found for {label_key}. Skipping (leaving as zeros).")
            continue

        # Get transform
        tform = transform_dict[label_key]

        # Convert numpy -> SimpleITK
        sitk_img = _sitk.GetImageFromArray(image_np)
        
        # Apply Resampling
        # Note: We use the image itself as the reference for size/spacing
        resampler = _sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_img)
        resampler.SetTransform(tform)
        resampler.SetInterpolator(_sitk.sitkLinear) # Use sitkNearestNeighbor for masks
        resampler.SetDefaultPixelValue(0) # Value for pixels outside the image
        
        warped_sitk = resampler.Execute(sitk_img)
        
        # Convert back to numpy
        transformed_stack[i, :, :] = _sitk.GetArrayFromImage(warped_sitk)
        valid_labels.append(label_key)

    return transformed_stack, valid_labels



def CombineFrames(transformed_stack, labels_list, n_wavelengths):
    """
    Averages registered frames belonging to the same wavelength index.

    Parameters:
    -----------
    transformed_stack : np.ndarray
        3D array [N_valid, Y, X] output from ApplyAllTransforms.
    labels_list : list
        List of labels matching transformed_stack (e.g., ["S0_Frame0", "S1_Frame0"...]).
    n_wavelengths : int
        The expected number of unique wavelengths (frames per sweep).

    Returns:
    --------
    combined_cube : np.ndarray
        3D array [Nwavelengths, Y, X] containing the averaged data.
    """
    N, h, w = transformed_stack.shape
    
    # Initialize accumulators
    # sum_cube stores the pixel sums
    sum_cube = np.zeros((n_wavelengths, h, w), dtype=np.float32)
    # count_cube stores how many frames contributed to each pixel
    count_cube = np.zeros((n_wavelengths, h, w), dtype=np.float32)

    print("Combining frames by averaging...")

    for i, label in enumerate(labels_list):
        # Parse the label to get the frame index
        # Assuming label format "S{sweep}_Frame{wav}" or similar
        # We look for the part after the underscore or explicitly find the Wav index
        
        # Regex to find the index associated with the "Frame" part
        # Looks for "Frame" followed by digits
        match = re.search(r'Frame(\d+)', label)
        if match:
            wav_idx = int(match.group(1))
        else:
            # Fallback: if you used a custom list like ['Blue', 'Red']
            # We have to map string -> index manually. 
            # For now, assuming "FrameX" format as per previous steps.
            print(f"Error parsing label {label}. Skipping.")
            continue
            
        if wav_idx >= n_wavelengths:
            continue

        # Add current frame to accumulator
        sum_cube[wav_idx] += transformed_stack[i]
        count_cube[wav_idx] += 1

    # Divide by count to get average
    # Avoid division by zero where no frames were present
    with np.errstate(divide='ignore', invalid='ignore'):
        combined_cube = sum_cube / count_cube
        combined_cube[count_cube == 0] = 0 # Handle empty frames

    return combined_cube
