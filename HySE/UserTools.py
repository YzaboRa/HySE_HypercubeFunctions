"""

Functions that represent tools for the user (plotting, saving, help, etc.)

"""


import numpy as np
import cv2
import os
from datetime import datetime
from scipy.signal import savgol_filter, find_peaks
import matplotlib
from matplotlib import pyplot as plt
import imageio
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import inspect
import copy
import re
import ast


from matplotlib.widgets import Slider, Button

matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"





def FindPlottingRange(array, **kwargs):
	"""
	Function that helps finding a reasonable range for plotting 
	Helpful when the data has several pixels with abnormally high/low values such that 
	the automatic range does not allow to visualise data (frequent after normalisation, when
	some areas of the image are dark)

	Input:
		- array (to plot)
		- kwargs:
			- std_range (default 3)
			- std_max_range 
			- std_min_range
			- Help

	Output:
		- m: Min value for plotting (vmin=m)
		- M: Max value for plitting (vmax=M)

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(FindPlottingRange))
		return 0,0

	std_range = kwargs.get('std_range', 3)
	std_max_range = kwargs.get('std_max_range', std_range)
	std_min_range = kwargs.get('std_min_range', std_range)

	array_flat = array.flatten()
	array_sorted = np.sort(array_flat)    
	mean = np.average(array_sorted)
	std = np.std(array_sorted)
	MM = mean + std_max_range*std
	mm = mean - std_min_range*std
	return mm, MM


def find_closest(arr, val):
	"""
	Function that finds the index in a given array whose value is the closest to a provided value

	Input:
		- arr: Array from which the index will be pulled fromGetDark_WholeVideo
		- val: Value to match as closely as possible

	Outout:
		- idx: Index from the provided array whose value is the closest to provided value

	"""
	idx = np.abs(arr - val).argmin()
	return idx

# def find_closest(array, value):
# 	""" Finds the index of the element in the array closest to the given value. """
# 	array = np.asarray(array)
# 	idx = (np.abs(array - value)).argmin()
# 	return idx


def wavelength_to_rgb(wavelength, gamma=0.8, **kwargs):

	'''This converts a given wavelength of light to an 
	approximate RGB color value. The wavelength must be given
	in nanometers in the range from 380 nm through 750 nm
	(789 THz through 400 THz).
	Based on code by Dan Bruton
	http://www.physics.sfasu.edu/astro/color/spectra.html

	Input:
		- wavelength (in nm)
		- gamma (default 0.8): transparancy value

	Return:
		- (R, G, B) value corresponding to the colour of the wavelength
	'''

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(wavelength_to_rgb))
		return (0,0,0)

	wavelength = float(wavelength)
	if wavelength >= 380 and wavelength <= 440:
		attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
		R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
		G = 0.0
		B = (1.0 * attenuation) ** gamma
	elif wavelength >= 440 and wavelength <= 490:
		R = 0.0
		G = ((wavelength - 440) / (490 - 440)) ** gamma
		B = 1.0
	elif wavelength >= 490 and wavelength <= 510:
		R = 0.0
		G = 1.0
		B = (-(wavelength - 510) / (510 - 490)) ** gamma
	elif wavelength >= 510 and wavelength <= 580:
		R = ((wavelength - 510) / (580 - 510)) ** gamma
		G = 1.0
		B = 0.0
	elif wavelength >= 580 and wavelength <= 645:
		R = 1.0
		G = (-(wavelength - 645) / (645 - 580)) ** gamma
		B = 0.0
	elif wavelength >= 645 and wavelength <= 750:
		attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
		R = (1.0 * attenuation) ** gamma
		G = 0.0
		B = 0.0
	else:
		R = 0.0
		G = 0.0
		B = 0.0
	R *= 255
	G *= 255
	B *= 255
#     return (int(R), int(G), int(B))
	return (R/256.0, G/256.0, B/256.0)



class MidpointNormalize(matplotlib.colors.Normalize):
	def __init__(self, vmin, vmax, midpoint=0, clip=False):
		self.midpoint = midpoint
		matplotlib.colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
		normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
		normalized_mid = 0.5
		x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
		return np.ma.masked_array(np.interp(value, x, y))




def PlotCoRegistered(im_static, im_shifted, im_coregistered, **kwargs):
	"""
	Function that produces a figure showing the co-registration of a given shifted image.

	Input:
		- im_static
		- im_shifted
		- im_coregistered
		- kwargs:
			- Help
			- ShowPlot (default False)
			- SavePlot (default False)
			- SavingPathWithName (default ''): If Saving figure, indicate the path where to save it
				Include the full name and '.png'.

	Output:
		- (Plots figure)

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(PlotCoRegistered))

	SavingPathWithName = kwargs.get('SavingPathWithName', '')
	SavePlot = kwargs.get('SavePlot', False)
	ShowPlot = kwargs.get('ShowPlot', False)

	images_diff_0 = np.subtract(im_shifted.astype('float64'), im_static.astype('float64'))
	images_diff_0_avg = np.average(np.abs(images_diff_0))
#     images_diff_0_std = np.std(np.abs(images_diff_0))
	images_diff_cr = np.subtract(im_coregistered.astype('float64'), im_static.astype('float64'))
	images_diff_cr_avg = np.average(np.abs(images_diff_cr))
#     images_diff_cr_std = np.average(np.std(images_diff_cr))

#     mmm, MMM = 0, 255
	mmm = min(np.amin(im_static), np.amin(im_shifted), np.amin(im_coregistered))
	MMM = max(np.amax(im_static), np.amax(im_shifted), np.amax(im_coregistered))
	mm0, MM0 = FindPlottingRange(images_diff_0)
	mm, MM = FindPlottingRange(images_diff_cr)

	norm = MidpointNormalize(vmin=mm0, vmax=MM0, midpoint=0)
	cmap = 'RdBu_r'

	m, M = FindPlottingRange(im_static)
	fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,7))
	im00 = ax[0,0].imshow(im_static, cmap='gray',vmin=m, vmax=M)
	ax[0,0].set_title('Static Image')
	divider = make_axes_locatable(ax[0,0])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im00, cax=cax, orientation='vertical')

	m, M = FindPlottingRange(im_shifted)
	im01 = ax[0,1].imshow(im_shifted, cmap='gray',vmin=m, vmax=M)
	ax[0,1].set_title('Shifted Image')
	divider = make_axes_locatable(ax[0,1])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im01, cax=cax, orientation='vertical')

	im02 = ax[0,2].imshow(images_diff_0, cmap=cmap, norm=norm)
	ax[0,2].set_title(f'Difference (no registration)\n avg {images_diff_0_avg:.2f}')
	divider = make_axes_locatable(ax[0,2])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im02, cax=cax, orientation='vertical')

	m, M = FindPlottingRange(im_static)
	im10 = ax[1,0].imshow(im_static, cmap='gray',vmin=m, vmax=M)
	ax[1,0].set_title('Static Image')
	divider = make_axes_locatable(ax[1,0])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im10, cax=cax, orientation='vertical')

	m, M = FindPlottingRange(im_coregistered)
	im11 = ax[1,1].imshow(im_coregistered, cmap='gray',vmin=m, vmax=M)
	ax[1,1].set_title('Coregistered Image')
	divider = make_axes_locatable(ax[1,1])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im11, cax=cax, orientation='vertical')

	im12 = ax[1,2].imshow(images_diff_cr, cmap=cmap, norm=norm)
	ax[1,2].set_title(f'Difference (with registration)\n avg {images_diff_cr_avg:.2f}')
	divider = make_axes_locatable(ax[1,2])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im12, cax=cax, orientation='vertical')

	## Add grid to help see changes in images
	(YY, XX) = im_static.shape
	xm, ym = int(XX/2), int(YY/2)
	xmm, ymm = int(xm/2), int(ym/2)
	x_points = [xmm, xm, xm+xmm, 3*xmm]
	y_points = [ymm, ym, ym+ymm, 3*ymm]
	for i in range(0,3):
		for j in range(0,2):
			ax[j,i].set_xticks([])
			ax[j,i].set_yticks([])
			for k in range(0,4):
				ax[j,i].axvline(x_points[k], c='limegreen', ls='dotted')
				ax[j,i].axhline(y_points[k], c='limegreen', ls='dotted')

	plt.tight_layout()
	if SavePlot:
		if '.png' not in SavingPathWithName:
			SavingPathWithName = SavingPathWithName+'_CoRegistration.png'
		print(f'Saving figure @ {SavingPathWithName}')
		# print(f'   Set SavingPathWithName=\'path\' to set saving path')
		plt.savefig(f'{SavingPathWithName}')
	if ShowPlot:
		plt.show()
	else:
		plt.close()



def PlotHypercube(Hypercube, **kwargs):
	"""
	Function to plot the hypercube.
	Input
		- Hypercube (np array)
		- kwargs:
			- Wavelengths: List of sorted wavelengths (for titles colours, default black)
			- Masks
			- SavePlot: (default False)
			- SavingPathWithName: Where to save the plot if SavePlot=True
			- ShowPlot: (default True)
			- SameScale (default False)
			- vmax
			- Help

	Output:
		- Figure (4x4, one wavelength per subfigure)
		Saved:
		if SavePlot=True:
			Figure

	"""

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(PlotHypercube))
		return 0

	Wavelengths = kwargs.get('Wavelengths')
	if Wavelengths is None:
		Wavelengths = [0]
		print("Input 'Wavelengths' list for better plot")

	SavePlot = kwargs.get('SavePlot', False)
	SavingPathWithName = kwargs.get('SavingPathWithName')
	if SavingPathWithName is None:
		SavingPathWithName = ''
		if SavePlot:
			print(f'SavePlot is set to True. Please input a SavingPathWithName')

	ShowPlot = kwargs.get('ShowPlot', True)
	SameScale = kwargs.get('SameScale', False)
	vmax = kwargs.get('vmax')
	if vmax is not None:
		SameScale = True
	Masks = kwargs.get('Masks')
	if Masks is None:
		MaskPlots = False
	else:
		MaskPlots = True

	Wavelengths_sorted = np.sort(Wavelengths)


	NN, YY, XX = Hypercube.shape

	nn = 0
	# plt.close()
	fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
	for j in range(0,4):
		for i in range(0,4):
			if nn<NN:
				if Wavelengths[0]==0:
					wav = 0
					RGB = (0,0,0) ## Set title to black if no wavelength input
				else:
					wav = Wavelengths_sorted[nn]
					RGB = wavelength_to_rgb(wav)

				if MaskPlots:
					array = Hypercube[nn,:,:]
					if len(Masks.shape)==2:
						mask = Masks
					elif len(Masks.shape)==3:
						mask = Masks[nn,:,:]
					else:
						print(f'Masks shape error:  {Masks.shape}')
						return 0
					ArrayToPlot = np.ma.array(array, mask=mask)
				else:
					ArrayToPlot = Hypercube[nn,:,:]

				if SameScale:
					if vmax is None:
						ax[j,i].imshow(ArrayToPlot, cmap='gray', vmin=0, vmax=np.amax(Hypercube))
					else:
						ax[j,i].imshow(ArrayToPlot, cmap='gray', vmin=0, vmax=vmax)
				else:
					ax[j,i].imshow(ArrayToPlot, cmap='gray', vmin=0, vmax=np.average(ArrayToPlot)*3)
				if wav==0:
					ax[j,i].set_title(f'{nn} wavelength', c=RGB)
				else:
					ax[j,i].set_title(f'{wav} nm', c=RGB)
				ax[j,i].set_xticks([])
				ax[j,i].set_yticks([])
				nn = nn+1
			else:
				ax[j,i].set_xticks([])
				ax[j,i].set_yticks([])

	plt.tight_layout()
	if SavePlot:
		if '.png' not in SavingPathWithName:
			SavingPathWithName = SavingPathWithName+'_Hypercube.png'
		plt.savefig(f'{SavingPathWithName}')
	if ShowPlot:
		plt.show()
	else:
		plt.close()



# def MakeHypercubeVideo(Hypercube, SavingPathWithName, **kwargs):
# 	'''
# 	Function that saves a mp4 video of the hypercube
# 	Input:
# 		- Hypercube
# 		- SavingPathWithName
# 		- kwargs:
# 			- fps: frame rate for the video (default 10)
# 			- Normalise = False. If true, self normalise
# 			- Help
# 	Output:
# 		Saved:
# 			mp4 video
# 	'''
# 	Help = kwargs.get('Help', False)
# 	if Help:
# 		print(inspect.getdoc(MakeHypercubeVideo))
	
# 	fps = kwargs.get('fps', 10)
# 	Normalise = kwargs.get('Normalise', False)

# 	if Normalise:
# 		Hypercube_ToSave = Hypercube/np.nanmax(Hypercube)
# 		print(f'Normalising Hypercube by itself')
# 	else:
# 		Hypercube_ToSave = Hypercube

# 	(NN, YY, XX) = Hypercube.shape
# 	if '.mp4' not in SavingPathWithName:
# 		SavingPathWithName = SavingPathWithName+'.mp4'

# 	out = cv2.VideoWriter(SavingPathWithName, cv2.VideoWriter_fourcc(*'mp4v'), fps, (XX, YY), False)
# 	for i in range(NN):
# 		data = Hypercube[i,:,:].astype('uint8')
# 		out.write(data)
# 	out.release()


# def MakeHypercubeVideo(Hypercube, SavingPathWithName, **kwargs):
# 	'''
# 	Function that saves a mp4 video of the hypercube.
# 	Safely handles np.nan values by converting them to black (0).
	
# 	Input:
# 		- Hypercube: A 3D numpy array (Frames, Y, X) of float dtype, may contain nans.
# 		- SavingPathWithName
# 		- kwargs:
# 			- fps: frame rate for the video (default 10)
# 			- Help

# 	Output:
# 		Saved:
# 			mp4 video
# 	'''
# 	Help = kwargs.get('Help', False)
# 	if Help:
# 		print(inspect.getdoc(MakeHypercubeVideo))
# 		return

# 	fps = kwargs.get('fps', 10)

# 	# --- Step 1: Handle NaNs and Scale the entire Hypercube ---
# 	# It ensures that the brightest pixel across all frames becomes 255,
# 	# and all other pixels are scaled relative to that.
	
# 	# Find the maximum value in the cube, ignoring NaNs
# 	max_val = np.nanmax(Hypercube)
	
# 	# Avoid division by zero if the hypercube is all zeros or NaNs
# 	if max_val == 0:
# 		max_val = 1.0 

# 	# Scale the float data to the 0-255 range
# 	scaled_hypercube = (Hypercube / max_val) * 255.0

# 	# Now, replace any remaining NaN values with 0 (black)
# 	scaled_hypercube[np.isnan(scaled_hypercube)] = 0
	
# 	# It is now safe to convert the entire cube to uint8
# 	hypercube_to_save = scaled_hypercube.astype('uint8')

# 	# --- Step 2: Write the Video ---
# 	(NN, YY, XX) = hypercube_to_save.shape
# 	if not SavingPathWithName.endswith('.mp4'):
# 		SavingPathWithName += '.mp4'

# 	# The 'False' argument at the end means we are creating a grayscale video
# 	out = cv2.VideoWriter(SavingPathWithName, cv2.VideoWriter_fourcc(*'mp4v'), fps, (XX, YY), False)
	
# 	for i in range(NN):
# 		# Each frame is now a clean uint8 array, ready to be written
# 		out.write(hypercube_to_save[i, :, :])
		
# 	out.release()
# 	print(f"Video saved to {SavingPathWithName}")


def ApplyMask(Hypercube, Mask, **kwargs):
	"""
	Function that applies a mask to all frames in a 3D array
	
	Inputs:
		- Hypercube
		- Mask
		- kwargs:
			- Help
			- FillValue = np.nan : What to replace masked pixels by
			
	Outputs:
		- MaskedHypercube
	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(ApplyMask))
		return 0
	
	FillValue = kwargs.get('FillValue', np.nan)
	
	(N, Y, X) = Hypercube.shape
	Hypercube_Masked = np.zeros(Hypercube.shape)
	for c in range(0,N):
		frame = copy.deepcopy(Hypercube[c,:,:])
		frame[Mask] = FillValue
		Hypercube_Masked[c,:,:] = frame
	return Hypercube_Masked


def MakeHypercubeVideo(Hypercube, SavingPathWithName, **kwargs):
	'''
	Function that saves a mp4 video of the hypercube.
	Safely handles np.nan values by converting them to black (0).

	Input:
		- Hypercube: A 3D numpy array (Frames, Y, X) of float dtype, may contain nans.
		- SavingPathWithName
		- kwargs:
			- Help
			- fps = 10: frame rate for the video
			- Mask (2D array): If indicated, mask all frames with this mask
			

	Output:
		Saved:
			mp4 video
	'''
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(MakeHypercubeVideo))
		return

	fps = kwargs.get('fps', 10)
	
	Mask = kwargs.get('Mask')
	(N, Y, X) = Hypercube.shape
	Hypercube_Masked = np.zeros(Hypercube.shape)
	if Mask is not None:
		Hypercube_Masked = ApplyMask(Hypercube, Mask)
	else:
		Hypercube_Masked = Hypercube

	# --- Step 1: Handle NaNs and Scale the entire Hypercube ---
	# It ensures that the brightest pixel across all frames becomes 255,
	# and all other pixels are scaled relative to that.

	# Find the maximum value in the cube, ignoring NaNs
	max_val = np.nanmax(Hypercube_Masked)

	# Avoid division by zero if the hypercube is all zeros or NaNs
	if max_val == 0:
		max_val = 1.0 

	# Scale the float data to the 0-255 range
	scaled_hypercube = (Hypercube_Masked / max_val) * 255.0

	# Now, replace any remaining NaN values with 0 (black)
	scaled_hypercube[np.isnan(scaled_hypercube)] = 0

	# It is now safe to convert the entire cube to uint8
	hypercube_to_save = scaled_hypercube.astype('uint8')

	# --- Step 2: Write the Video ---
	(NN, YY, XX) = hypercube_to_save.shape
	if not SavingPathWithName.endswith('.mp4'):
		SavingPathWithName += '.mp4'

	# The 'False' argument at the end means we are creating a grayscale video
	out = cv2.VideoWriter(SavingPathWithName, cv2.VideoWriter_fourcc(*'mp4v'), fps, (XX, YY), False)

	for i in range(NN):
		# Each frame is now a clean uint8 array, ready to be written
		out.write(hypercube_to_save[i, :, :])

	out.release()
	print(f"Video saved to {SavingPathWithName}")


def PlotDark(Dark):
	"""
	Function that plots the dark reference for inspection

	Inputs:
		- Dark

	Outputs:
		- (plot figure)

	"""
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
	im = ax.imshow(Dark, cmap='gray')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax.set_title(f'Dark \navg: {np.average(Dark):.2f}, min: {np.nanmin(Dark):.2f}, max: {np.nanmax(Dark):.2f}')
	plt.show()




### Macbeth colour charts

def GetPatchPos(Patch1_pos, Patch_size_x, Patch_size_y, Image_angle, **kwargs):
	"""
	Function that estimates the position of each patch in the macbeth chart. 
	It identifies a corner in the patch, and defines a square region that should sit within the regions of the patch.
	If the squares do not fit nicely in the patches, play with the different parameters.
	To be used in testing/calibratin datasets done by imaging a standard macbeth colourchart.
	The output is designed to be checked with PlotPatchesDetection() and used with GetPatchesSpectrum() functions.

	Inputs:
		- Patch1_pos [y0,x0]: Coordinates of patch 1 (brown, corner)
		- Patch_size_x: Estimate (in pixels) of the spacing between patches in the x axis
		- Patch_size_y: Estimate (in pixels) of the spacing between patches in the y axis
		- Image_angle: angle (in degrees) of the chart in the image
		- kwargs:
			- Help

	Outputs:
		- Positions: Array containing the coordinates for each of the 30 patches 

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetPatchPos))

	Positions = []
	[y0, x0] = Patch1_pos
	Image_angle_rad = Image_angle*np.pi/180
	index = 0
	for j in range(0,5):
		y0s = y0 -  j*Patch_size_y*np.cos(Image_angle_rad) #j*Patch_size -
		x0s = x0 + j*Patch_size_x*np.sin(Image_angle_rad)
		for i in range(0,6):
			x = x0s - Patch_size_x*np.cos(Image_angle_rad)*i
			y = y0s - Patch_size_x*np.sin(Image_angle_rad)*i
			# if (j==0 and i==5):
			# 	y = y-15
			# 	x = x+10
			Positions.append([index, x, y])
			index +=1
	return np.array(Positions)

def rough_radial_scaling(x, y, x_c, y_c, k1, k2, diag2):
	dx = x - x_c
	dy = y - y_c
	r2 = (dx**2 + dy**2) / diag2  # normalize r^2
	scale = 1 / (1 + k1 * r2 + k2 * r2**2)
	return scale

def GetPatchPos_WithDistortion(Patch1_pos, Patch_size_x, Patch_size_y, Image_angle, **kwargs):
	"""
	Estimates patch positions on a Macbeth chart with rough barrel distortion compensation.
	Ensures all coordinates remain in the image coordinate system (positive values).

	Inputs:
		- Patch1_pos: [y0, x0] coordinate of patch 1 (top-left brown) in pixels
		- Patch_size_x/y: average patch spacing (pixels)
		- Image_angle: chart tilt in degrees (positive = counterclockwise)
		- kwargs:
			- k1, k2: barrel distortion coefficients (default 0)
			- ImageShape: (height, width) in pixels [required]
			- Help: prints this docstring

	Output:
		- Positions: array of [index, x, y] for each of the 30 patches (in distorted image coordinates)
	"""

	
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetPatchPos_WithDistortion))
		return 0

	k1 = kwargs.get('k1', 0.0)
	k2 = kwargs.get('k2', 0.0)
	ImageShape = kwargs.get('ImageShape', None)
	if ImageShape is None:
		raise ValueError("You must supply ImageShape=(height, width) for radial correction")

	if k1==0.0 and k2==0.0:
		Positions = []
		[y0, x0] = Patch1_pos
		Image_angle_rad = Image_angle*np.pi/180
		index = 0
		for j in range(0,5):
			y0s = y0 -  j*Patch_size_y*np.cos(Image_angle_rad) #j*Patch_size -
			x0s = x0 + j*Patch_size_x*np.sin(Image_angle_rad)
			for i in range(0,6):
				x = x0s - Patch_size_x*np.cos(Image_angle_rad)*i
				y = y0s - Patch_size_x*np.sin(Image_angle_rad)*i
				# if (j==0 and i==5):
				# 	y = y-15
				# 	x = x+10
				Positions.append([index, x, y])
				index +=1
	else:
		h_px, w_px = ImageShape
		x_c = w_px / 2
		y_c = h_px / 2

		Positions = []
		[y0, x0] = Patch1_pos
		angle_rad = Image_angle * np.pi / 180
		index = 0
		
		diag2 = w_px**2 + h_px**2

		for j in range(5):  # rows
			for i in range(6):  # cols
				# Estimate raw offset from patch 1
				dx = -i * Patch_size_x * np.cos(angle_rad) + j * Patch_size_x * np.sin(angle_rad)
				dy = -i * Patch_size_x * np.sin(angle_rad) - j * Patch_size_y * np.cos(angle_rad)
				x_guess = x0 + dx
				y_guess = y0 + dy

				# Apply local scale adjustment for barrel distortion
	#             scale = rough_radial_scaling(x_guess, y_guess, x_c, y_c, k1, k2)
				scale = rough_radial_scaling(x_guess, y_guess, x_c, y_c, k1, k2, diag2)
				dx_adj = dx * scale
				dy_adj = dy * scale

				x = x0 + dx_adj
				y = y0 + dy_adj
				# if j == 0 and i == 5:
				# 	y -= 15
				# 	x += 10

				# Ensure coordinates remain positive
				x = max(0, min(x, w_px - 1))
				y = max(0, min(y, h_px - 1))

				Positions.append([index, x, y])
				index += 1

	return np.array(Positions)

def GetPatchesSpectrum(Hypercube, Sample_size, Positions, CropCoordinates, **kwargs):
	"""
	Function that extracts the average spectrum in each patch region, as defined by the output from the GetPatchPos() function.

	Inputs:
		- Hypercube
		- Sample_size: size of a patch
		- Positions: Positions of each patch, output from GetPatchPos()
		- CropCoordingates: For the full image
		- kwargs:
			- Help

	Output:
		- Spectra: Array containing the average spectra for all patches

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetPatchesSpectrum))

	Spectrum = []
	(NN, YY, XX) = Hypercube.shape
	for n in range(0,NN):
		im_sub = Hypercube[n,:,:]
		Intensities = GetPatchesIntensity(im_sub[CropCoordinates[2]:CropCoordinates[3], CropCoordinates[0]:CropCoordinates[1]], Sample_size, Positions)
		Spectrum.append(Intensities)
	return np.array(Spectrum)


def GetPatchesIntensity(Image, Sample_size, PatchesPositions):
	"""
	Functino used in GetPatchesSpectrum() to calculate the average value (for a single wavelength/image) for all patches

	Inputs:
		- Image (slide of the hypercube, single wavelength)
		- Sample_size: estimated size of a patch (smaller than real patch to avoid unwanted pixels)
		- PatchesPositions: Positions of each patches, output of GetPatchPos()

	Outputs:
		- Intensities: Array size 30 containing the average intensity for this given image/wavelenght for all patches

	"""
	N = len(PatchesPositions)
	Intensities = []
	for n in range(0,N):
		nn = PatchesPositions[n,0]
		x0, y0 = PatchesPositions[n,1], PatchesPositions[n,2]
		xs, xe  = int(x0-Sample_size/2), int(x0+Sample_size/2)
		ys, ye  = int(y0-Sample_size/2), int(y0+Sample_size/2)
		im_sub = Image[ys:ye, xs:xe]
		val = np.nanmean(im_sub)
		std = np.nanstd(im_sub)
		Intensities.append([nn, val, std])       
	return np.array(Intensities)


def PlotPatchesDetection(macbeth, Positions, Sample_size):
	"""
	Function that plots the automatic patch position estimates over an image of the macbeth chart (from the data)
	Use this to make sure that the patches have been properly identified and that all the pixels included 
	are indeed part of the patch, to avoid corrupted spectra

	Inputs:
		- macbeth: monochromatic image of the macbeth chart (from the data, same dimensions)
		- Positions: Positions of each patches, output of GetPatchPos()
		- Sample_size: estimated size of a patch (smaller than real patch to avoid unwanted pixels)

	Outputs:
		- (plots figure)


	"""
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
	im = ax.imshow(macbeth, cmap='gray')
	for i in range(0, 30):  
		ax.scatter(Positions[i,1], Positions[i,2], color='cornflowerblue', s=15)
		ax.text(Positions[i,1]-10, Positions[i,2]-8, f'{Positions[i,0]+1:.0f}', color='red')
		area = patches.Rectangle((Positions[i,1]-Sample_size/2, Positions[i,2]-Sample_size/2), Sample_size, Sample_size, edgecolor='none', facecolor='cornflowerblue', alpha=0.4)
		ax.add_patch(area)
	plt.tight_layout()
	plt.show()



# def psnr(img1, img2, **kwargs):
# 	"""

# 	Function that computes the peak signal to noise ratio (PSNR) between two images
# 	Used to calculate how closely data matches a reference (spectra)

# 	Inputs:
# 		- img1
# 		- img2
# 		- kwargs:
# 			- Help

# 	Outputs:
# 		- psnr

# 	"""
# 	Help = kwargs.get('Help', False)
# 	if Help:
# 		print(inspect.getdoc(psnr))

# 	mse = np.mean(np.square(np.subtract(img1,img2)))
# 	if mse==0:
# 		return np.Inf
# 	max_pixel = 1 #255.0
# 	psnr = 20 * math.log10(max_pixel / np.sqrt(mse)) 
# #     psnr = 20 * math.log10(max_pixel) - 10 * math.log10(mse)  
# 	return psnr


def psnr(img1, img2, data_range=1.0, **kwargs):
	"""
	Function that computes the peak signal to noise ratio (PSNR) between two images
	Used to calculate how closely data matches a reference (spectra)

	Inputs:
		- img1
		- img2
		- kwargs:
			- Help

	Outputs:
		- psnr
	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(psnr))

	# mse = np.mean((img1 - img2) ** 2)
	mse = np.mean(np.square(np.subtract(img1,img2)))
	if mse == 0:
		# return float('inf')
		return np.Inf
	return 20 * np.log10(data_range / np.sqrt(mse))

def CompareSpectra(Wavelengths_sorted, GroundTruthWavelengths, GroundTruthSpectrum, **kwargs):
	"""
	Function that outputs a grout truth reference spectra of the same size as the data 
	Allows to plot both on the same x data list. 
	Assumes that the ground truth has more wavelengths than the dataset

	Inputs:
		- Wavelengths_sorted: list of wavelengths (data)
		- GroundTruthWavelengths: list of wavelengths (ground truth/reference)
		- GroundTruthSpectrum: Spectra of the ground truth/reference (same length as GroundTruthWavelengths)
		- kwargs:
			- Help

	Output:
		- Comparable_GroundTruthSpectrum


	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(CompareSpectra))

	ComparableSpectra = []
	for i in range(0,len(Wavelengths_sorted)):
		wav=Wavelengths_sorted[i]
		index=find_closest(GroundTruthWavelengths, wav)
		ComparableSpectra.append(GroundTruthSpectrum[index])
	return np.array(ComparableSpectra)


# def CompareSpectra(wavelengths_target, wavelengths_source, spectrum_source):
# 	"""
# 	Interpolates a source spectrum to match the target wavelengths.
# 	This is necessary to compare spectra sampled at different wavelengths.


# 	"""
# 	return np.interp(wavelengths_target, wavelengths_source, spectrum_source)


def correlation_coefficient(spec1, spec2):
	"""
	Function that calculates the Pearson correlation coefficient between two 1D arrays. 

	Inputs:
		- spec1
		- spec2

	Outputs
		- Correlation
	"""
	correlation = np.corrcoef(spec1, spec2)[0, 1]
	return np.abs(correlation)

def spectral_angle_mapper(spec1, spec2, eps=1e-8, **kwargs):
	"""
	Function that calculates the Spectral Angle Mapper (SAM) between two 1D arrays.

	Inputs:
		- spec1
		- spec2
		- eps = 1e-8 (epsilon, to avoid dividing by 0)
		- kwargs
			- Help
			- Radians = False. If true, outputs angle in radians instead of degrees
	
	Outputs:
		- Angle (degrees).
	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(spectral_angle_mapper))
		return
	Radians = kwargs.get('Radians', False)
	dot_product = np.dot(spec1, spec2)
	norm_spec1 = np.linalg.norm(spec1)
	norm_spec2 = np.linalg.norm(spec2)
	# Adding epsilon to avoid division by zero
	angle_rad = np.arccos(dot_product / (norm_spec1 * norm_spec2 + eps))
	if Radians:
		angle = angle_rad
	else:
		angle = np.degrees(angle_rad)
	return angle





def PlotPatchesSpectra(PatchesSpectra_All, Wavelengths_sorted_All, MacBethSpectraData, MacBeth_RGB, Name, **kwargs):
	'''
	Function to plot the spectra extracted from the patches of macbeth colour chart

	Inputs:
		- PatchesSpectra_All: an array, or a list of arrays. Each array is expected of the shape
			(Nwavelengths (16), Npatches (30), 3). Uses the output of the GetPatchesSpectrum()
			function.
		- Wavelengths_sorted: list of sorted wavelengths
		- MacBethSpectraData: Ground truth spectra for the macbeth patches
		- MacBeth_RGB: MacBeth RBG values for each patch (for plotting)
		- Name: Name of the dataset (for saving)

	- kwargs:
		- Help: print this info
		- SavingPath: If indicated, saves the figure at the indicated path
		- Metric = 'PSNR' : Indicates which metric to use to asses the quality of the spectrum
			Accepted metrics: 
			'PSNR', 
			'AvgCorrelation' (Pearson correlation coefficient), 
			'SAM' (Spectral Angle Mapper)
		- ChosenMethod (0). If more than one set of spectra provided, determines which
			of those (the 'method') has its metric indicated for each path.
		- PlotLabels: What label to put for each provided set of spectra. If not indicated
			a generic 'option 1', 'option 2' etc will be used
		- WhitePatchNormalise (True). Normalises all spectral by the spectra of the white patch
		- ClipYScale (True): Clip the y range of the plots to [-0.05,1.05]
		- ClipXScale (True)
		- XScale ([450,670])
		- KNORM 
		- NORMPATCHES
		- Colours

	Outputs:
		- (plots figure)

	Metric Priority: If multiple metrics are set to True, PSNR is used. If PSNR is False
	but AvgCorrelation is True, Correlation is used. If both are False, SAM is used.
	'''
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(PlotPatchesSpectra))
		return

	# --- Argument and Kwarg parsing ---
	SavingPath = kwargs.get('SavingPath')
	SaveFig = SavingPath is not None

	ClipYScale = kwargs.get('ClipYScale', True)
	ClipXScale = kwargs.get('ClipXScale', True)
	XScale = kwargs.get('XScale', [450, 670])

	ChosenMethod = kwargs.get('ChosenMethod', 0)
	WhitePatchNormalise = kwargs.get('WhitePatchNormalise', True)
	Metric = kwargs.get('Metric', 'PSNR')
	KNORM = kwargs.get('KNORM', [])
	NORMPATCHES = kwargs.get('NORMPATCHES')
	
	Colours = kwargs.get('Colours')
	

	metric_name = None
	metric_func = None
	metric_format = None
	# Priority: PSNR > Correlation > SAM
	if Metric=='PSNR':
		print(f'Using PSNR')
		metric_name = 'PSNR'
		metric_func = HySE.psnr
		metric_format = '{:.2f} dB'
	elif Metric=='AvgCorrelation':
		print(f'Using Average Correlation')
		metric_name = 'Correlation'
		metric_func = HySE.correlation_coefficient
		metric_format = '{:.3f}'
	elif Metric=='SAM':
		print(f'Using Spectral Angle Mapper')
		metric_name = 'SAM'
		metric_func = HySE.spectral_angle_mapper
		metric_format = '{:.2f}°'
	else:
		print(f'Metric ({Metric}) not accepted ! Use PSNR, AvgCorrelation or SAM')

	CalculateMetric = metric_name is not None
	all_patches_metric_values = [] # For calculating the average for the suptitle

	if Colours is not None:
		PlotColours=Colours
	else:
		PlotColours = ['limegreen', 'cornflowerblue', 'orange', 'red', 'darkblue', 'cyan', 'magenta']

	# --- Data preparation ---
	if not isinstance(PatchesSpectra_All, list):
		PatchesSpectra_All = [PatchesSpectra_All]

	PlotLabels = kwargs.get('PlotLabels')
	if PlotLabels is None:
		print(f'Indicate PlotLabels for more descriptive plot')
		PlotLabels = [f'Option {i+1}' for i in range(len(PatchesSpectra_All))]
		
	
#     AllWavs = Wavelengths_sorted_All
	flattened_list = [wavelength for sublist in Wavelengths_sorted_All for wavelength in sublist]
	AllWavs = np.array(flattened_list)
		
	WavelengthRange_start = np.round(int(np.amin(AllWavs)) / 10.0) * 10
	WavelengthRange_end = np.round(np.amax(AllWavs) / 10.0) * 10
	print(f'Wavelength range: {WavelengthRange_start} : {WavelengthRange_end}')

	idx_min_gtruth = HySE.find_closest(MacBethSpectraData[:, 0], WavelengthRange_start)
	idx_max_gtruth = HySE.find_closest(MacBethSpectraData[:, 0], WavelengthRange_end)

	Nwhite = 8 - 1 # Index for the white patch (patch #8)
	White_truth = MacBethSpectraData[idx_min_gtruth:idx_max_gtruth, Nwhite + 1]
	GroundTruthWavelengths = MacBethSpectraData[idx_min_gtruth:idx_max_gtruth, 0]


	# --- Plotting ---
	fig, ax = plt.subplots(nrows=5, ncols=6, figsize=(14, 12))
	for i in range(6):
		for j in range(5):
			patchN = i + j * 6
			
			color = (MacBeth_RGB[patchN, 0] / 255, MacBeth_RGB[patchN, 1] / 255, MacBeth_RGB[patchN, 2] / 255)

			GroundTruthSpectrum = MacBethSpectraData[idx_min_gtruth:idx_max_gtruth, patchN + 1]
			if WhitePatchNormalise:
				GroundTruthSpectrumN = np.divide(GroundTruthSpectrum, White_truth)
			else:
				GroundTruthSpectrumN = GroundTruthSpectrum

			ax[j, i].plot(GroundTruthWavelengths, GroundTruthSpectrumN, color='black', lw=4, label='Truth')

			metric_vals_per_method = []

			for k in range(len(PatchesSpectra_All)):
				PatchesSpectra = PatchesSpectra_All[k]
				Wavelengths_sorted = Wavelengths_sorted_All[k]
				
				WavelengthRange_start_sub = np.round(int(np.amin(Wavelengths_sorted)) / 10.0) * 10
				WavelengthRange_end_sub = np.round(np.amax(Wavelengths_sorted) / 10.0) * 10
#                 print(f'Wavelength range: {WavelengthRange_start} : {WavelengthRange_end}')

				idx_min_gtruth_sub = HySE.find_closest(MacBethSpectraData[:, 0], WavelengthRange_start_sub)
				idx_max_gtruth_sub = HySE.find_closest(MacBethSpectraData[:, 0], WavelengthRange_end_sub)

				Nwhite = 8 - 1 # Index for the white patch (patch #8)
				White_truth_sub = MacBethSpectraData[idx_min_gtruth_sub:idx_max_gtruth_sub, Nwhite + 1]
				GroundTruthWavelengths_sub = MacBethSpectraData[idx_min_gtruth_sub:idx_max_gtruth_sub, 0]
				
				GroundTruthSpectrum_sub = MacBethSpectraData[idx_min_gtruth_sub:idx_max_gtruth_sub, patchN + 1]
			
					
				if WhitePatchNormalise:
					GroundTruthSpectrumN_sub = np.divide(GroundTruthSpectrum_sub, White_truth_sub)
				else:
					GroundTruthSpectrumN_sub = GroundTruthSpectrum_sub
			
				
				if WhitePatchNormalise:
					spectra_WhiteNorm = np.divide(PatchesSpectra[:, patchN, 1], PatchesSpectra[:, Nwhite, 1])
				else:
					spectra_WhiteNorm = PatchesSpectra[:, patchN, 1]
					
				if k in KNORM:
					if NORMPATCHES is None:
						KL_Norm = np.average(GroundTruthSpectrumN_sub)
					else:
						NORMPATCHES
#                     print(f'For patch {patchN}, {PlotLabels[k]}: divide by 1/{KL_Norm}')
					spectra_WhiteNorm = spectra_WhiteNorm*KL_Norm
#                 print(len(Wavelengths_sorted))
#                 print(len(spectra_WhiteNorm))
				ax[j, i].plot(Wavelengths_sorted, spectra_WhiteNorm, '.-', c=PlotColours[k], label=PlotLabels[k])

				if CalculateMetric:
					# Interpolate ground truth to match our measurement wavelengths for comparison
					GT_comparable = HySE.CompareSpectra(Wavelengths_sorted, GroundTruthWavelengths_sub, GroundTruthSpectrumN_sub)
					metric_val = metric_func(GT_comparable, spectra_WhiteNorm)
					metric_vals_per_method.append(metric_val)

			# --- Axis and Title Formatting ---
			if ClipYScale:
				ax[j, i].set_ylim(-0.05, 1.05)
			if ClipXScale:
				ax[j, i].set_xlim(XScale[0], XScale[1])

			# Set title for the white patch specifically
			if patchN == Nwhite:
				ax[j, i].set_title(f'Patch {patchN+1} - white', color='black', fontsize=10)
				ax[j, i].legend(fontsize=7, loc='lower center')
			# Set title for all other patches
			else:
				if CalculateMetric:
					# Determine best method (max for PSNR/Corr, min for SAM)
					if metric_name == 'SAM':
						best_val_pos = np.argmin(metric_vals_per_method)
					else:
						best_val_pos = np.argmax(metric_vals_per_method)

					selected_val_str = metric_format.format(metric_vals_per_method[ChosenMethod])
					best_val_str = metric_format.format(metric_vals_per_method[best_val_pos])

					title_text = (f'Patch {patchN+1}\n'
								  f'Selected {metric_name} = {selected_val_str}\n'
								  f'Best: {best_val_str} ({PlotLabels[best_val_pos]})')
					ax[j, i].set_title(title_text, color=color, fontsize=8)
					all_patches_metric_values.append(metric_vals_per_method[ChosenMethod])
				else:
					ax[j, i].set_title(f'Patch {patchN+1}', color=color, fontsize=10)

			# --- Axis Labels ---
			if j == 4:
				ax[j, i].set_xlabel('Wavelength [nm]')
			else:
				ax[j, i].xaxis.set_ticklabels([])

			if i == 0:
				ax[j, i].set_ylabel('Normalized intensity')
			elif ClipYScale: # Only remove labels if y-scale is clipped and not the first column
				ax[j, i].yaxis.set_ticklabels([])

	# --- Final Figure Formatting and Saving ---
	suptitle_text = f'Spectra for {Name}'
	if CalculateMetric:
#         print(all_patches_metric_values)
		avg_metric = np.mean(np.abs(all_patches_metric_values))
		avg_metric_str = metric_format.format(avg_metric)
		suptitle_text += (f' - Selected Method: {PlotLabels[ChosenMethod]}\n'
						  f'Average {metric_name} for all patches: {avg_metric_str}')

	plt.suptitle(suptitle_text, fontsize=12)
	plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

	if SaveFig:
		plt.savefig(f'{SavingPath}_Patches.png', dpi=300)
	plt.show()


def GetPatchesMetrics(PatchesSpectra_All, Wavelengths_sorted, MacBethSpectraData, MacBeth_RGB, Name, **kwargs):
	'''
	Function to compute how close a reconstructed spectra is to a ground truth spectra.
	Designed to look at each patches in a Macbeth chart.
	Uses three metrics:
		- PSNR
		- Pearson Correlation
		- SAM


	Inputs:
		- PatchesSpectra_All: an array, or a list of arrays. Each array is expected of the shape
			(Nwavelengths (16), Npatches (30), 3). Uses the output of the GetPatchesSpectrum()
			function.
		- Wavelengths_sorted: list of sorted wavelengths
		- MacBethSpectraData: Ground truth spectra for the macbeth patches
		- kwargs:
			- Help: print this info
			- ChosenMethod (0). If more than one set of spectra provided, determines which
				of those (the 'method') has its metric indicated for each path.
			- WhitePatchNormalise (True). Normalises all spectral by the spectra of the white patch

	Outputs:
		- all_patches_PSNR
		- all_patches_Correlation
		- all_patches_SAM
		- (printout of the averages)
	

	'''
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(PlotPatchesSpectra))
		return 0,0,0

	ChosenMethod = kwargs.get('ChosenMethod', 0)
	WhitePatchNormalise = kwargs.get('WhitePatchNormalise', True)

	all_patches_PSNR = [] 
	all_patches_Correlation = [] 
	all_patches_SAM = [] 
		
	if not isinstance(PatchesSpectra_All, list):
		PatchesSpectra_All = [PatchesSpectra_All]
		

	WavelengthRange_start = np.round(int(np.amin(Wavelengths_sorted)) / 10.0) * 10
	WavelengthRange_end = np.round(np.amax(Wavelengths_sorted) / 10.0) * 10
	# print(f'Wavelength range: {WavelengthRange_start} : {WavelengthRange_end}')

	idx_min_gtruth = find_closest(MacBethSpectraData[:, 0], WavelengthRange_start)
	idx_max_gtruth = find_closest(MacBethSpectraData[:, 0], WavelengthRange_end)

	Nwhite = 8 - 1 # Index for the white patch (patch #8)
	White_truth = MacBethSpectraData[idx_min_gtruth:idx_max_gtruth, Nwhite + 1]
	GroundTruthWavelengths = MacBethSpectraData[idx_min_gtruth:idx_max_gtruth, 0]

	for i in range(6):
		for j in range(5):
			patchN = i + j * 6
			color = (MacBeth_RGB[patchN, 0] / 255, MacBeth_RGB[patchN, 1] / 255, MacBeth_RGB[patchN, 2] / 255)
			
			GroundTruthSpectrum = MacBethSpectraData[idx_min_gtruth:idx_max_gtruth, patchN + 1]
			if WhitePatchNormalise:
				GroundTruthSpectrumN = np.divide(GroundTruthSpectrum, White_truth)
			else:
				GroundTruthSpectrumN = GroundTruthSpectrum
							
			for k in range(len(PatchesSpectra_All)):
				PatchesSpectra = PatchesSpectra_All[k]
				if WhitePatchNormalise:
					spectra_WhiteNorm = np.divide(PatchesSpectra[:, patchN, 1], PatchesSpectra[:, Nwhite, 1])
				else:
					spectra_WhiteNorm = PatchesSpectra[:, patchN, 1]
				
				# Interpolate ground truth to match our measurement wavelengths for comparison
				GT_comparable = CompareSpectra(Wavelengths_sorted, GroundTruthWavelengths, GroundTruthSpectrumN)

				if patchN!=Nwhite: ## Ignore white patch since it's used to normalise

					## PSNR
					metric_val = psnr(GT_comparable, spectra_WhiteNorm)
					all_patches_PSNR.append(metric_val)

					## Correlation
					metric_val = correlation_coefficient(GT_comparable, spectra_WhiteNorm)
					all_patches_Correlation.append(metric_val)

					## SAM
					metric_val = spectral_angle_mapper(GT_comparable, spectra_WhiteNorm)
					all_patches_SAM.append(metric_val)

		
	avg_PSNR = np.mean(all_patches_PSNR)
	avg_Correlation = np.mean(np.abs(all_patches_Correlation))
	avg_SAM = np.mean(np.abs(all_patches_SAM))

	std_PSNR = np.std(all_patches_PSNR)
	std_Correlation = np.std(np.abs(all_patches_Correlation))
	std_SAM = np.std(np.abs(all_patches_SAM))
	
	print(f'Average Values for each Metric:')
	print(f'   PSNR:        {avg_PSNR:.2f} +/- {std_PSNR:.2f} dB')
	print(f'   Correlation: {avg_Correlation:.3f} +/- {std_Correlation:.3f}')
	print(f'   SAM:         {avg_SAM:.2f} +/- {std_SAM:.2f} °')

	return all_patches_PSNR, all_patches_Correlation, all_patches_SAM
	



def SelectMacbethPatches(image_frame):
	"""
	Interactive tool to define 30 Macbeth chart patches on an endoscopic image.
	
	Features:
	- Click to add points (up to 30).
	- Double-click an existing point to 'pick it up', move mouse to position, 
	  click again to drop.
	- Slider to adjust the sample area size (transparent square overlay).
	- 'Done' button to close and return data.

	Parameters:
		image_frame (numpy.ndarray): The 2D or 3D image data.

	Returns:
		tuple: 
			- positions (numpy.ndarray): Shape (30, 3) containing [n_patch, x, y].
			- sample_size (int): The side length of the square sampling area.
	"""
	
	class MacbethSelector:
		def __init__(self, ax, img_shape):
			self.ax = ax
			self.points = []      # List of (x, y)
			self.labels = []      # List of text objects
			self.rects = []       # List of Rectangle objects
			self.dots = []        # List of plot objects (visual dots)
			self.dragging = None  # Index of point currently being moved
			self.img_h, self.img_w = img_shape[:2]
			
			# Initial sample size
			self.sample_size = 10
			
			self.canvas = ax.figure.canvas

		def add_point(self, x, y):
			if len(self.points) >= 30:
				return
			
			idx = len(self.points) + 1
			self.points.append([x, y])
			
			# 1. Plot center dot
			dot, = self.ax.plot(x, y, 'r+', markersize=10, markeredgewidth=2)
			self.dots.append(dot)
			
			# 2. Add Label
			lbl = self.ax.text(x, y - 15, str(idx), color='yellow', 
							   fontsize=12, ha='center', fontweight='bold')
			self.labels.append(lbl)
			
			# 3. Add Square (centered)
			offset = self.sample_size / 2
			rect = Rectangle((x - offset, y - offset), 
							 self.sample_size, self.sample_size,
							 linewidth=1, edgecolor='cyan', 
							 facecolor=(0, 1, 1, 0.3)) # Transparent cyan
			self.ax.add_patch(rect)
			self.rects.append(rect)
			
			self.update_title()
			self.canvas.draw_idle()

		def update_point_position(self, idx, x, y):
			# Update data
			self.points[idx] = [x, y]
			
			# Update visuals
			self.dots[idx].set_data([x], [y])
			self.labels[idx].set_position((x, y - 15))
			
			offset = self.sample_size / 2
			self.rects[idx].set_xy((x - offset, y - offset))
			
			self.canvas.draw_idle()

		def update_square_size(self, val):
			self.sample_size = int(val)
			offset = self.sample_size / 2
			
			for i, rect in enumerate(self.rects):
				x, y = self.points[i]
				rect.set_width(self.sample_size)
				rect.set_height(self.sample_size)
				rect.set_xy((x - offset, y - offset))
			
			self.canvas.draw_idle()

		def get_closest_point(self, x, y, threshold=50):
			if not self.points:
				return None
			
			dists = np.sqrt(np.sum((np.array(self.points) - np.array([x, y]))**2, axis=1))
			min_idx = np.argmin(dists)
			
			if dists[min_idx] < threshold:
				return min_idx
			return None

		def update_title(self):
			self.ax.set_title(f"Patches identified: {len(self.points)}/30\n"
							  "Click to add. Double-click point to grab/move.")

	# --- Setup Figure and Widgets ---
	fig, ax = plt.subplots(figsize=(10, 8))
	plt.subplots_adjust(bottom=0.2) # Make room for controls
	
	ax.imshow(image_frame, cmap='gray' if image_frame.ndim == 2 else None)
	ax.axis('off')
	
	selector = MacbethSelector(ax, image_frame.shape)
	selector.update_title()

	# Slider
	ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
	slider = Slider(ax_slider, 'Sample Size', 1, 100, valinit=10, valstep=1)
	
	# Done Button
	ax_button = plt.axes([0.81, 0.01, 0.1, 0.075])
	btn = Button(ax_button, 'Done')

	# --- Event Callbacks ---

	def on_click(event):
		# Ignore clicks outside main axes (like on the slider/button)
		if event.inaxes != ax:
			return

		# Double Click: Pick up a point
		if event.dblclick:
			idx = selector.get_closest_point(event.xdata, event.ydata)
			if idx is not None:
				selector.dragging = idx
				# Visual feedback: change label color to red while dragging
				selector.labels[idx].set_color('red')
				selector.canvas.draw_idle()
		
		# Single Click
		else:
			# If currently dragging, drop it
			if selector.dragging is not None:
				selector.labels[selector.dragging].set_color('yellow')
				selector.dragging = None
				selector.canvas.draw_idle()
			# Otherwise, add new point
			else:
				selector.add_point(event.xdata, event.ydata)

	def on_move(event):
		if event.inaxes == ax and selector.dragging is not None:
			selector.update_point_position(selector.dragging, event.xdata, event.ydata)

	def on_slider_update(val):
		selector.update_square_size(val)

	def on_done(event):
		plt.close(fig)

	# Wiring events
	fig.canvas.mpl_connect('button_press_event', on_click)
	fig.canvas.mpl_connect('motion_notify_event', on_move)
	slider.on_changed(on_slider_update)
	btn.on_clicked(on_done)

	plt.show(block=True)

	# --- Format Output ---
	# Construct (30, 3) array: [patch_num, x, y]
	positions = np.zeros((30, 3))
	for i, (x, y) in enumerate(selector.points):
		# Ensure we don't exceed 30 in the output array if user clicked too many times
		if i >= 30: break 
		positions[i] = [i + 1, x, y]

	return positions, selector.sample_size

def PlotMixingMatrix(MixingMatrix, Wavelengths, Title, SavingPath):
	Nwavs_ = len(Wavelengths)
	Nims, Nwavs = MixingMatrix.shape
	if Nwavs!=Nwavs:
		print(f'Error: There number of wavelengths does not match between Wavelengths list {Nwavs_} and the mixing matrix {MixingMatrix.shape}')
	xx = [i for i in range(0,Nwavs)]
	bin_labels = [f'im {i+1}' for i in range(0,Nims)]

	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
	ax.imshow(MixingMatrix, cmap='magma')
	ax.set_xticks(xx)
	ax.set_yticks(xx)
	ax.set_xticklabels(Wavelengths, rotation=90)
	ax.set_yticklabels(bin_labels)

	ax.set_xlabel(f'Individual Wavelengths [nm]')
	ax.set_ylabel(f'Combined Wavelengths [nm]')
	
	ax.set_title(f'Mixing Matrix - {Title}')
	plt.tight_layout()
#     plt.savefig(SavingPath)
	plt.show()




def process_spectral_file(filename):
	"""
	Reads a text file with spectral data and organizes it into a NumPy array.

	The function extracts a list of method names and a dictionary of patch data,
	then assembles them into an array where each row corresponds to a patch and
	each column corresponds to a method.

	Args:
		filename (str): The path to the input text file.

	Returns:
		A tuple containing:
		- list: A list of method names.
		- np.ndarray: A 2D NumPy array of the data (Patches x Methods).
	
	Raises:
		FileNotFoundError: If the specified file does not exist.
		ValueError: If method names or patch data cannot be found in the file.
	"""
	methods = []
	patch_data = {}

	with open(filename, 'r') as f:
		for line in f:
			line = line.strip()
			# Find the line containing the list of methods
			if line.startswith("['") and line.endswith("']"):
				# Use ast.literal_eval for safely parsing the list string
				methods = ast.literal_eval(line)
			# Find all lines that start with 'Patch'
			elif line.startswith('Patch'):
				# Use a regular expression to extract the patch number and the values
				match = re.search(r"Patch (\d+): \[(.*)\]", line)
				if match:
					patch_num = int(match.group(1))
					# Split the values string and convert each item to a float
					values = [float(val.replace(' ', '').strip("'")) for val in match.group(2).split(',')]
					patch_data[patch_num] = values

	if not methods or not patch_data:
		raise ValueError("Could not find method names or patch data in the file.")

	# Determine the dimensions of the final array from the collected data
	num_patches = max(patch_data.keys())
	num_methods = len(methods)

	# Create an empty NumPy array filled with NaN (Not a Number)
	# This helps identify if any patches were missing in the source file
	result_array = np.full((num_patches, num_methods), np.nan)

	# Populate the array using the patch number to determine the row index
	for patch_num, values in patch_data.items():
		# Subtract 1 from patch_num for 0-based array indexing
		result_array[patch_num - 1] = values

	return methods, result_array



def RestoreNaNs(OriginalMixed, UnmixedData):
	"""
	Restores NaN values to the unmixed data based on the spatial locations 
	of NaNs in the original mixed hypercube.
	
	Parameters:
	-----------
	OriginalMixed : np.ndarray
		3D array [N_wavelengths, Y, X] containing the original NaNs.
	UnmixedData : np.ndarray
		3D array [N_endmembers, Y, X] output from the unmixing algorithm 
		where NaNs were converted to 0.
		
	Returns:
	--------
	restored_mixed : np.ndarray
		The mixed array with guaranteed NaNs in the background.
	restored_unmixed : np.ndarray
		The unmixed array with NaNs restored to the background.
	"""
	# Make copies to ensure we don't accidentally modify the originals in-place
	restored_mixed = np.copy(OriginalMixed)
	restored_unmixed = np.copy(UnmixedData)
	
	# 1. Create a 2D Spatial Mask
	# If a pixel is NaN in ANY wavelength, we consider it a background pixel.
	# Evaluating along axis=0 collapses the spectral dimension, leaving a (Y, X) boolean mask.
	spatial_nan_mask = np.isnan(OriginalMixed).any(axis=0)
	
	# 2. Apply the mask to restore NaNs
	# Using NumPy broadcasting, we apply the 2D spatial mask across all spectral/endmember channels
#     restored_mixed[:, spatial_nan_mask] = np.nan
	restored_unmixed[:, spatial_nan_mask] = np.nan
#     restored_mixed, 
	return restored_unmixed



