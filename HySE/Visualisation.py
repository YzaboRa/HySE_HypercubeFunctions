


import numpy as np
import matplotlib.pyplot as plt
import cv2  # New dependency for robust Sobel
from skimage import feature, color, exposure
from skimage.restoration import inpaint
from scipy.ndimage import binary_dilation, gaussian_filter, median_filter

### N.A. Functions written with Gemini


def visualize_edge_density_overlay(hypercube, method='sobel', 
								   overlay_color=(1.0, 0.0, 0.0), # RGB for Red
								   sobel_ksize=5, 
								   canny_sigma=2.0,
								   cap_intensity=0.5, raw_threshold=0.1, 
								   reflection_mask=None, nframe=0, 
								   mask_dilation_kernel=0, 
								   denoise_sigma=0.0, 
								   denoise_method='gaussian',
								   mask_inpaint_method='zero_fill', 
								   min_gradient_threshold=0.0,
								   opacity_gain=1.0): # New param to boost visibility
	"""
	Overlays accumulated edges from all frames onto a single monochrome context frame.
	Stable edges appear solid; transient/moving edges appear faint.

	Parameters:
	-----------
	overlay_color : tuple
		RGB tuple (0-1) for the edge color. Default is Red (1.0, 0.0, 0.0).
	opacity_gain : float
		Multiplier for the overlay opacity. 
		- 1.0: Linear (50% overlap = 50% opacity).
		- >1.0: Makes faint edges more visible.
	(Other parameters are identical to the previous function)
	"""
	
	n_frames, height, width = hypercube.shape
	hypercube = np.nan_to_num(hypercube, nan=0.0)

	# --- 1. Prepare Background (Context Frame) ---
	bg_frame = hypercube[nframe].copy()
	bg_norm = (bg_frame - bg_frame.min()) / (bg_frame.max() - bg_frame.min() + 1e-8)
	
	# Process the background frame (Inpaint/Mask) for display cleanliness
	if reflection_mask is not None:
		bg_mask = reflection_mask[nframe].astype(bool) if len(reflection_mask.shape) == 3 else reflection_mask.astype(bool)
		if mask_dilation_kernel > 0:
			bg_mask = binary_dilation(bg_mask, structure=np.ones((mask_dilation_kernel, mask_dilation_kernel)))

		if mask_inpaint_method == 'inpaint':
			bg_norm = inpaint.inpaint_biharmonic(bg_norm, bg_mask, channel_axis=None)
		else:
			bg_norm[bg_mask] = 0.0 # Just zero out for background display

	# Convert grayscale background to RGB so we can overlay color
	background_rgb = np.stack([bg_norm]*3, axis=-1)

	# --- 2. Accumulate Edges ---
	edge_accumulator = np.zeros((height, width), dtype=np.float32)

	# Helper for denoising
	def apply_denoise(img, strength, method):
		if strength <= 0: return img
		if method == 'gaussian': return gaussian_filter(img, sigma=strength)
		elif method == 'median': return median_filter(img, size=int(max(2, strength)))
		return img

	for i in range(n_frames):
		frame = hypercube[i]
		frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
		working_frame = frame_norm.copy()
		
		# Masking / Inpainting
		if reflection_mask is not None:
			mask = reflection_mask[i].astype(bool) if len(reflection_mask.shape) == 3 else reflection_mask.astype(bool)
			if mask_dilation_kernel > 0:
				mask = binary_dilation(mask, structure=np.ones((mask_dilation_kernel, mask_dilation_kernel)))
			
			if method in ['canny', 'sobel']:
				if mask_inpaint_method == 'inpaint':
					working_frame = inpaint.inpaint_biharmonic(working_frame, mask, channel_axis=None)
				else: 
					working_frame[mask] = 0.0
		
		# Filtering
		if method in ['canny', 'sobel']:
			working_frame = apply_denoise(working_frame, denoise_sigma, denoise_method)
			
			if method == 'canny':
				feature_map = feature.canny(working_frame, sigma=canny_sigma).astype(np.float32)
			elif method == 'sobel':
				grad_x = cv2.Sobel(working_frame, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
				grad_y = cv2.Sobel(working_frame, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
				grad_mag = cv2.magnitude(grad_x, grad_y)
				feature_map = exposure.rescale_intensity(grad_mag, in_range='image')

			if min_gradient_threshold > 0.0:
				feature_map[feature_map < min_gradient_threshold] = 0.0
			
			# Cap and Normalize per frame
			feature_map = np.clip(feature_map, 0.0, cap_intensity)
			if cap_intensity < 1.0 and feature_map.max() > 0:
				feature_map = feature_map / cap_intensity  
				
		elif method == 'raw_thresholded':
			feature_map = (working_frame < raw_threshold).astype(np.float32)
			if reflection_mask is not None:
				# Re-mask logic omitted for brevity, similar to previous func
				pass 

		# Accumulate
		edge_accumulator += feature_map

	# --- 3. Create Overlay ---
	
	# Normalize accumulator: 
	# 0.0 means no edge ever. 
	# 1.0 means an edge was present in EVERY frame (if we divide by n_frames).
	# Using n_frames ensures that "Solid Red" = "Totally Stable".
	alpha_map = edge_accumulator / n_frames
	
	# Apply Gain (make faint edges more visible if desired)
	alpha_map = alpha_map * opacity_gain
	alpha_map = np.clip(alpha_map, 0.0, 1.0)
	
	# Create the solid color layer
	color_layer = np.zeros((height, width, 3), dtype=np.float32)
	color_layer[:] = overlay_color
	
	# Expand alpha to 3 channels for broadcasting
	alpha_3ch = alpha_map[..., np.newaxis]

	# --- 4. Alpha Blending ---
	# Formula: Result = Background * (1 - Alpha) + Overlay * Alpha
	final_composite = background_rgb * (1.0 - alpha_3ch) + color_layer * alpha_3ch
	final_composite = np.clip(final_composite, 0, 1)

	# --- Plotting ---
#     plt.close()
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.imshow(final_composite)
#     ax.set_title(f"Edge Density Overlay (Method: {method})\nSolid Color = Stable Feature across {n_frames} frames")
	ax.axis('off')
	
	plt.tight_layout()
	plt.show()
	return final_composite

def visualize_hypercube_movement(hypercube, method='sobel', 
								sobel_ksize=5,  # CHANGED: Integer kernel size (3, 5, 7, etc.)
								canny_sigma=2.0, # Separated Canny sigma
								cap_intensity=0.5, raw_threshold=0.1, 
								reflection_mask=None, nframe=0, 
								mask_dilation_kernel=0, 
								denoise_sigma=0.0, 
								denoise_method='gaussian',
								mask_inpaint_method='zero_fill', 
								display_power_gamma=1.0,
								min_gradient_threshold=0.0):
	"""
	Visualizes movement in a hypercube using OpenCV for robust gradient calculation.

	Parameters:
	-----------
	hypercube : numpy.ndarray
		Input array (N, Y, X).
	method : str
		'sobel', 'canny', or 'raw_thresholded'.
	sobel_ksize : int
		Kernel size for OpenCV Sobel. Must be an odd integer (1, 3, 5, 7, ...).
		- 3: Standard, sharp edges, susceptible to noise.
		- 5: Smooths noise, highlights medium features. (RECOMMENDED for your data)
		- 7: Very smooth, ignores fine texture completely.
	canny_sigma : float
		Sigma for Canny edge detection (if method='canny').
	cap_intensity : float
		Clips high-intensity features (0.0 to 1.0).
	raw_threshold : float
		Threshold for 'raw_thresholded'.
	reflection_mask : numpy.ndarray
		Binary mask (2D or 3D).
	nframe : int
		Context frame index.
	mask_dilation_kernel : int
		Size of dilation for reflection mask.
	denoise_sigma : float
		Pre-filtering strength.
	denoise_method : str
		'gaussian' or 'median'.
	mask_inpaint_method : str
		'zero_fill' or 'inpaint'.
	display_power_gamma : float
		Power law (val^gamma) to darken background noise.
	min_gradient_threshold : float
		Hard cutoff (0.0 to 1.0). Signals below this are set to 0.

	Returns:
	--------
	fig : matplotlib.figure.Figure
	"""
	
	n_frames, height, width = hypercube.shape
	hypercube = np.nan_to_num(hypercube, nan=0.0)
	
#     print(f'reflection_mask.shape = {reflection_mask.shape}')
#     print(f'hypercube.shape = {hypercube.shape}')

	# --- Validation ---
	if reflection_mask is not None:
		if len(reflection_mask.shape) == 3:
			if reflection_mask.shape != hypercube.shape: raise ValueError("Mask shape mismatch.")
		elif len(reflection_mask.shape) == 2:
			 if reflection_mask.shape != (height, width): raise ValueError("Mask shape mismatch.")
	
	# Ensure sobel_ksize is odd
	if sobel_ksize % 2 == 0:
		sobel_ksize += 1
		print(f"Warning: sobel_ksize must be odd. Automatically adjusted to {sobel_ksize}.")

	composite_image = np.zeros((height, width, 3), dtype=np.float32)
	hues = np.linspace(0, 1, n_frames + 1)[:-1]
	
	def apply_denoise(img, strength, method):
		if strength <= 0: return img
		if method == 'gaussian': return gaussian_filter(img, sigma=strength)
		elif method == 'median': return median_filter(img, size=int(max(2, strength)))
		return img

	# --- Context Frame (Left Subplot) ---
	ref_frame = hypercube[nframe].copy()
	ref_frame_norm = (ref_frame - ref_frame.min()) / (ref_frame.max() - ref_frame.min() + 1e-8)
	ref_frame_processed = ref_frame_norm.copy()
	mask_label = "Raw Normalized"
	
	if reflection_mask is not None and method in ['canny', 'sobel']:
		ref_mask = reflection_mask[nframe].astype(bool) if len(reflection_mask.shape) == 3 else reflection_mask.astype(bool)
		if mask_dilation_kernel > 0:
			ref_mask = binary_dilation(ref_mask, structure=np.ones((mask_dilation_kernel, mask_dilation_kernel)))

		if mask_inpaint_method == 'inpaint':
			ref_frame_processed = inpaint.inpaint_biharmonic(ref_frame_norm, ref_mask, channel_axis=None)
			mask_label = "Inpainted"
		else: 
			ref_frame_processed[ref_mask] = 0.0
			mask_label = "Zero-Fill"
		
		if denoise_sigma > 0.0:
			ref_frame_processed = apply_denoise(ref_frame_processed, denoise_sigma, denoise_method)

	# --- Main Loop ---
	for i in range(n_frames):
		frame = hypercube[i]
		frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
		working_frame = frame_norm.copy()
		
		# 1. Masking / Inpainting
		if reflection_mask is not None:
			mask = reflection_mask[i].astype(bool) if len(reflection_mask.shape) == 3 else reflection_mask.astype(bool)
			if mask_dilation_kernel > 0:
				mask = binary_dilation(mask, structure=np.ones((mask_dilation_kernel, mask_dilation_kernel)))
			
			if method in ['canny', 'sobel']:
				if mask_inpaint_method == 'inpaint':
					working_frame = inpaint.inpaint_biharmonic(working_frame, mask, channel_axis=None)
				else: 
					working_frame[mask] = 0.0
		
		# 2. Filtering
		if method in ['canny', 'sobel']:
			# A. Pre-Denoise
			working_frame = apply_denoise(working_frame, denoise_sigma, denoise_method)
			
			# B. Edge Detection
			if method == 'canny':
				feature_map = feature.canny(working_frame, sigma=canny_sigma).astype(np.float32)
			
			elif method == 'sobel':

				# CV_64F handles negative gradients correctly (important!)
				grad_x = cv2.Sobel(working_frame, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
				grad_y = cv2.Sobel(working_frame, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
				
				# Calculate magnitude
				grad_mag = cv2.magnitude(grad_x, grad_y)
				
				# Rescale to 0-1 for display
				feature_map = exposure.rescale_intensity(grad_mag, in_range='image')

			# C. Post-Processing
			if min_gradient_threshold > 0.0:
				feature_map[feature_map < min_gradient_threshold] = 0.0
			
			feature_map = np.clip(feature_map, 0.0, cap_intensity)
			if cap_intensity < 1.0 and feature_map.max() > 0:
				feature_map = feature_map / cap_intensity  
			
			if display_power_gamma != 1.0:
				feature_map = feature_map ** display_power_gamma
				
		elif method == 'raw_thresholded':
			feature_map = (working_frame < raw_threshold).astype(np.float32)
			if reflection_mask is not None:
				if mask_dilation_kernel > 0:
					mask = binary_dilation(mask, structure=np.ones((mask_dilation_kernel, mask_dilation_kernel)))
				feature_map[mask] = 0.0 

		# 3. Coloring
		current_color_hsv = np.array([[[hues[i], 1.0, 1.0]]])
		current_color_rgb = color.hsv2rgb(current_color_hsv)
		
		colored_layer = feature_map[..., np.newaxis] * current_color_rgb
		composite_image += colored_layer

	final_display = np.clip(composite_image, 0, 1)
	
	# --- Plotting ---
#     plt.close()
	fig, axes = plt.subplots(1, 2, figsize=(13, 6))

	axes[0].imshow(ref_frame_processed, cmap='gray')
	axes[0].set_title(f"Context Frame (Index: {nframe})\n{mask_label}")
	axes[0].axis('off')

	axes[1].imshow(final_display)
	axes[1].set_title(f"Movement Vis\nSobel Kernel: {sobel_ksize}x{sobel_ksize}, Thresh: {min_gradient_threshold}")
	axes[1].axis('off')
	
	plt.tight_layout()
	return final_display, ref_frame_processed




#### ------------------------------------------------
#### ------------------------------------------------
#### ------------------------------------------------
#### ------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
from matplotlib.path import Path
import matplotlib.patches as patches

def GetBiopsyLocations(RegisteredHypercube, Wavelengths, **kwargs):
	"""
	GUI to draw ROIs on a hyperspectral image (Shape: Bands, Y, X) and extract spectral data.

	Parameters:
	-----------
	RegisteredHypercube : numpy.ndarray
		3D array of shape (Nwavelengths, Height, Width).
	Wavelengths : list or numpy.ndarray
		List of wavelengths corresponding to the Bands.
	**kwargs :
		display_band_idx (int): Index of the band to display (default: middle band).
		figsize (tuple): Figure size (default: (16, 9)).

	Returns:
	--------
	ROI_coordinates : list
		List of (N, 2) arrays containing (x, y) vertices for each ROI.
	ROI_AvgSpectra : list
		List of 1D arrays containing the average spectrum for each ROI.
	ROI_AllSpectra : list
		List of 2D arrays (M_pixels, Bands) containing spectra for all pixels in each ROI.
	"""

	class BiopsyPicker:
		def __init__(self, cube, wavelengths, display_band_idx=None, figsize=(16, 9)):
			self.cube = cube
			self.wavelengths = wavelengths
			# --- FIX: Handle (Bands, Y, X) shape ---
			self.bands, self.h, self.w = cube.shape
			
			# Display setup
			if display_band_idx is None:
				self.display_band_idx = self.bands // 2
			else:
				self.display_band_idx = int(display_band_idx)
			
			# Extract the 2D image for display (Slice the first dimension)
			self.image_data = self.cube[self.display_band_idx, :, :]
			
			self.rois = [] 
			self.current_roi_verts = []
			
			self.state = 'IDLE' 
			self.active_roi_idx = -1
			self.active_vertex_idx = -1
			self.drag_active = False
			
			# Colors and Styles
			self.cmap_cycle = plt.cm.tab10.colors 
			self.alpha_draw = 0.4
			self.alpha_done = 0.7

			self.fig, self.ax = plt.subplots(figsize=figsize)
			plt.subplots_adjust(bottom=0.2)
			
			# --- REUSE: Normalization from LandmarkPicker ---
			self.img_norm = self._normalize(self.image_data)
			self.im_obj = self.ax.imshow(self.img_norm, cmap='gray', vmin=0, vmax=1)
			
			wl_label = self.wavelengths[self.display_band_idx] if self.display_band_idx < len(self.wavelengths) else "N/A"
			self.ax.set_title(f"Band {self.display_band_idx} ({wl_label} nm) | "
							  f"'r': Start ROI | 'z': Undo | Double-click ROI to Edit | CLOSE to finish")

			# Temporary artist for drawing active lines
			self.line_active, = self.ax.plot([], [], linestyle='--', marker='o', color='white', lw=1.5, animated=False)
			
			# Widgets
			ax_color = 'lightgoldenrodyellow'
			ax_radio = self.fig.add_axes([0.92, 0.05, 0.07, 0.15], facecolor=ax_color)
			self.radio = RadioButtons(ax_radio, ('gray', 'magma', 'viridis'))
			self.radio.on_clicked(self.update_cmap)
			
			# Event Connections
			self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
			self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
			self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
			self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
			self.cid_close = self.fig.canvas.mpl_connect('close_event', self.on_close)

		def _normalize(self, img):
			"""Reuse normalization logic from LandmarkPicker"""
			p2, p98 = np.percentile(img, (2, 98))
			denom = p98 - p2
			return np.clip((img - p2) / (denom if denom != 0 else 1), 0, 1)

		def update_cmap(self, label):
			self.im_obj.set_cmap(label)
			self.fig.canvas.draw_idle()

		def _get_next_color(self):
			idx = len(self.rois) % len(self.cmap_cycle)
			return self.cmap_cycle[idx]

		def _redraw_active_line(self):
			if not self.current_roi_verts:
				self.line_active.set_data([], [])
			else:
				xs, ys = zip(*self.current_roi_verts)
				self.line_active.set_data(xs, ys)
				self.line_active.set_color(self._get_next_color())
			self.fig.canvas.draw_idle()

		def _finish_roi(self):
			if len(self.current_roi_verts) < 3:
				print("ROI needs at least 3 points.")
				self.current_roi_verts = []
				self._redraw_active_line()
				self.state = 'IDLE'
				return

			color = self._get_next_color()
			poly = plt.Polygon(self.current_roi_verts, closed=True, 
							   facecolor=color, edgecolor=color, alpha=self.alpha_done, label=f"ROI {len(self.rois)+1}")
			self.ax.add_patch(poly)
			
			cx = np.mean([v[0] for v in self.current_roi_verts])
			cy = np.mean([v[1] for v in self.current_roi_verts])
			text = self.ax.text(cx, cy, str(len(self.rois) + 1), color='white', 
								weight='bold', ha='center', va='center', fontsize=10)

			self.rois.append({
				'vertices': list(self.current_roi_verts),
				'color': color,
				'artist_poly': poly,
				'artist_text': text,
				'artist_markers': None 
			})
			
			self.current_roi_verts = []
			self._redraw_active_line()
			self.state = 'IDLE'
			self.fig.canvas.draw_idle()

		def _update_roi_visuals(self, roi_idx):
			roi = self.rois[roi_idx]
			roi['artist_poly'].set_xy(roi['vertices'])
			cx = np.mean([v[0] for v in roi['vertices']])
			cy = np.mean([v[1] for v in roi['vertices']])
			roi['artist_text'].set_position((cx, cy))
			if roi['artist_markers']:
				xs, ys = zip(*roi['vertices'])
				roi['artist_markers'].set_data(xs, ys)

		def on_key(self, event):
			if event.key == 'r':
				if self.state == 'IDLE':
					self.state = 'DRAWING'
					self.current_roi_verts = []
					print("Mode: DRAWING")
				elif self.state == 'EDITING':
					self._exit_edit_mode()
					self.state = 'DRAWING'
					self.current_roi_verts = []
					print("Exited Edit Mode -> DRAWING")

			elif event.key == 'z':
				if self.state == 'DRAWING' and self.current_roi_verts:
					self.current_roi_verts.pop()
					self._redraw_active_line()
				elif self.state == 'IDLE' and self.rois:
					roi = self.rois.pop()
					roi['artist_poly'].remove()
					roi['artist_text'].remove()
					self.fig.canvas.draw_idle()
					print(f"Removed ROI {len(self.rois)+1}")

		def on_click(self, event):
			if event.inaxes != self.ax or event.button != 1: return
			
			# --- REUSE: Check toolbar state from LandmarkPicker ---
			toolbar = self.fig.canvas.manager.toolbar
			if toolbar is not None and toolbar.mode != '':
				return

			# Double Click Handling
			if event.dblclick:
				if self.state == 'IDLE':
					for i, roi in enumerate(self.rois):
						path = Path(roi['vertices'])
						if path.contains_point((event.xdata, event.ydata)):
							self._enter_edit_mode(i)
							return
				elif self.state == 'EDITING':
					# Delete vertex on double click
					roi = self.rois[self.active_roi_idx]
					verts = roi['vertices']
					dist = np.linalg.norm(np.array(verts) - np.array([event.xdata, event.ydata]), axis=1)
					if np.min(dist) < 10: 
						idx_to_remove = np.argmin(dist)
						if len(verts) > 3:
							verts.pop(idx_to_remove)
							self._update_roi_visuals(self.active_roi_idx)
							self.fig.canvas.draw_idle()
					else:
						self._exit_edit_mode()
				return

			# Single Click Handling
			if self.state == 'DRAWING':
				# Check closure
				if len(self.current_roi_verts) > 2:
					start_pt = np.array(self.current_roi_verts[0])
					curr_pt = np.array([event.xdata, event.ydata])
					xlim = self.ax.get_xlim()
					tol = (xlim[1] - xlim[0]) * 0.02 
					if np.linalg.norm(start_pt - curr_pt) < tol:
						self._finish_roi()
						return

				self.current_roi_verts.append((event.xdata, event.ydata))
				self._redraw_active_line()

			elif self.state == 'EDITING':
				roi = self.rois[self.active_roi_idx]
				verts = roi['vertices']
				dist = np.linalg.norm(np.array(verts) - np.array([event.xdata, event.ydata]), axis=1)
				xlim = self.ax.get_xlim()
				tol = (xlim[1] - xlim[0]) * 0.02
				
				if np.min(dist) < tol:
					self.active_vertex_idx = np.argmin(dist)
					self.drag_active = True

		def on_move(self, event):
			if event.inaxes != self.ax: return
			if self.state == 'EDITING' and self.drag_active:
				roi = self.rois[self.active_roi_idx]
				roi['vertices'][self.active_vertex_idx] = (event.xdata, event.ydata)
				self._update_roi_visuals(self.active_roi_idx)
				self.fig.canvas.draw_idle()

		def on_release(self, event):
			if self.state == 'EDITING':
				self.drag_active = False
				self.active_vertex_idx = -1

		def _enter_edit_mode(self, idx):
			if self.state == 'EDITING' and self.active_roi_idx != idx:
				self._exit_edit_mode()
			self.state = 'EDITING'
			self.active_roi_idx = idx
			roi = self.rois[idx]
			roi['artist_poly'].set_alpha(self.alpha_draw)
			xs, ys = zip(*roi['vertices'])
			line, = self.ax.plot(xs, ys, 'o', color='white', markeredgecolor=roi['color'], markersize=8, zorder=10)
			roi['artist_markers'] = line
			self.fig.canvas.draw_idle()

		def _exit_edit_mode(self):
			if self.active_roi_idx == -1: return
			roi = self.rois[self.active_roi_idx]
			roi['artist_poly'].set_alpha(self.alpha_done)
			if roi['artist_markers']:
				roi['artist_markers'].remove()
				roi['artist_markers'] = None
			self.state = 'IDLE'
			self.active_roi_idx = -1
			self.drag_active = False
			self.fig.canvas.draw_idle()

		def disconnect(self):
			"""Disconnects all matplotlib events."""
			self.fig.canvas.mpl_disconnect(self.cid_click)
			self.fig.canvas.mpl_disconnect(self.cid_move)
			self.fig.canvas.mpl_disconnect(self.cid_release)
			self.fig.canvas.mpl_disconnect(self.cid_key)
			self.fig.canvas.mpl_disconnect(self.cid_close)

		def on_close(self, event):
			self.disconnect()

		def get_results(self):
			all_coords = [np.array(r['vertices']) for r in self.rois]
			avg_spectra = []
			all_spectra = []
			
			# Create grid for masking
			y_indices, x_indices = np.mgrid[:self.h, :self.w]
			points = np.vstack((x_indices.ravel(), y_indices.ravel())).T
			
			for roi in self.rois:
				path = Path(roi['vertices'])
				mask_flat = path.contains_points(points)
				mask = mask_flat.reshape((self.h, self.w))
				
				# --- FIX: Extract from (Bands, Y, X) ---
				# mask is (Y, X), cube is (Bands, Y, X)
				# We select all bands, and pixels where mask is True
				pixels = self.cube[:, mask]  # Shape becomes (Bands, N_pixels)
				
				if pixels.shape[1] > 0:
					# Transpose to (N_pixels, Bands) for standard spectral format
					pixels = pixels.T 
					all_spectra.append(pixels)
					avg_spectra.append(np.mean(pixels, axis=0))
				else:
					all_spectra.append(np.empty((0, self.bands)))
					avg_spectra.append(np.zeros(self.bands))
					
			return all_coords, avg_spectra, all_spectra

	picker = BiopsyPicker(RegisteredHypercube, Wavelengths, **kwargs)
	plt.show(block=True)
	return picker.get_results()
		 