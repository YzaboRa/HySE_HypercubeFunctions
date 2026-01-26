


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