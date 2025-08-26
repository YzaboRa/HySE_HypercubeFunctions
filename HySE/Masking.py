"""

Functions used to handle masks

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
import time
from tqdm import trange
import inspect
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"
from scipy.ndimage import median_filter

# import HySE_ImportData
import HySE.UserTools
# import HySE.ManipulateHypercube



def RemoveSpecularReflections_Frame(frame, **kwargs):
	"""
	Detects and removes specular reflections from an image frame.
	
	Inputs:
		- frame : Input 2D image (Y, X), float values (dark-subtracted, 0–~255).
		- kwargs:
			- Help
			- k = 3 : Threshold factor (default). Reflection mask is frame > mean + k*std.
				Lower k = more masking
			- Cutoff : When specified, use this value as threshold for specular reflections
			- NeighborhoodSize = 5 : When specified (default), used to compute median value around
				masked area and use a fill value
			- FillValue  : When specified, replace masked pixels by this value
	
	Outputs:
	
		- frame_corrected : Image with specular reflections replaced by local median values/fill value.
		- mask :  Binary mask of specular reflections (uint8, 0 or 1).
	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(ApplyTransform))
		return 0, 0
	FillValue = kwargs.get('FillValue')
	Cutoff = kwargs.get('Cutoff')
	
	# Ensure float array
	img = frame.astype(np.float32)
	
	# Step 1: Identify reflections
	mean_val = np.mean(img)
	std_val = np.std(img)
	if Cutoff is None:
		k = kwargs.get('k', 3)
		threshold = mean_val + k * std_val
	else:
		threshold = Cutoff
	mask = (img > threshold).astype(np.uint8)

	frame_corrected = img.copy()
	if FillValue is not None:
		frame_corrected[mask == 1] = FillValue
		
	else:
		NeighborhoodSize = kwargs.get('NeighborhoodSize', 5)
		median_img = median_filter(img, size=NeighborhoodSize)
		frame_corrected[mask == 1] = median_img[mask == 1]


	return frame_corrected, mask


def RemoveSpecularReflections(Frames, **kwargs):
	"""
	Detects and removes specular reflections from a set of frames (Nwav, Y, X).
	
	Inputs:
		- frames : Input frames (Nwav, Y, X), float values (dark-subtracted, 0–~255).
		- kwargs:
			- Help
			- k = 3 : Threshold factor (default). Reflection mask is frame > mean + k*std.
				Lower k = more masking
			- Cutoff : When specified, use this value as threshold for specular reflections
			- NeighborhoodSize = 5 : When specified (default), used to compute median value around
				masked area and use a fill value
			- FillValue  : When specified, replace masked pixels by this value
	
	Outputs:
	
		- MaskedFrames : Images with specular reflections replaced by local median values/fill 
			value for each frame.
		- AllMasks :  Binary masks of specular reflections (uint8, 0 or 1) for each frame.
	"""
	(Nwav, Y, X) = Frames.shape
	MaskedFrames = []
	AllMasks = []
	for n in range(0,Nwav):
		frame = Frames[n,:,:]
		frame_corrected, mask = RemoveSpecularReflections_Frame(frame, **kwargs)
		MaskedFrames.append(frame_corrected)
		AllMasks.append(mask)
	AllMasks = np.array(AllMasks)
	MaskedFrames = np.array(MaskedFrames)
	return MaskedFrames, AllMasks





def GetStandardMask(WhiteCalibration, **kwargs):
	"""
	OLD
	Returns a array (mask) that includes every pixel with a value below a threshold.
	Threshold can be indicated, or set automatically to 1.
	Best to use white calibration data for optimal mask

	Inputs:
	- WhiteCalibration (or image)
	- kwargs (optional)
		- threshold
		- Help

	Outputs:
	- Mask

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetStandardMask))
		return 0
	threshold = kwargs.get('threshold', 1)
	Calibration_avg = np.average(np.average(WhiteCalibration, axis=1), axis=1)
	max_idx = np.where(Calibration_avg==np.amax(Calibration_avg))[0][0]
	Mask = WhiteCalibration[max_idx, :,:] < threshold
	return Mask

def ConvertMaskToBinary(mask):
	"""
	OLD
	Invert mask (1-mask_value) and convert to binary format

	Inputs: 
	- Mask

	Output:
	- BinaryMask (inverted)

	"""
	mask_binary = 1-mask*1
	return mask_binary.astype('uint8')

def BooleanMaskOperation(bool_white, bool_wav):
	"""
	OLD
	Old function to handle boolean operation for masks

	"""
	bool_result = False
	if bool_white!=bool_wav:
		if bool_wav==1:
			bool_result = True
	return bool_result

# def TakeWavMaskDiff(mask_white, mask_shifted):
# 	vectorized_function = np.vectorize(BooleanMaskOperation)
# 	result = vectorized_function(mask_white, mask_shifted)
# 	return result

def CombineMasks(mask1, mask2):
	"""
	OLD
	Function that combines two maks (OR operator).
	A pixel that is masked in either (or both) of the masks will be masked in the final mask

	Inputs:
		- mask1
		- mask2

	Outputs:
		- combined_mask


	"""
	combined_mask = np.ma.mask_or(mask1, mask2)
	combined_mask = combined_mask*1
	combined_mask = mask.astype('uint8')
	return combined_mask
	

def GetMask(frame, **kwargs):
	'''
	OLD
	Inputs:
		- frame (2D array)
		- kwargs
			- LowCutoff: noise level, default 0.8
			- HighCutoff: specular reflection, default none
			- PlotMask: plotting masks and image, default False
			- Help

	Outputs:
		- Combined masks

	'''
	Help = kwargs.get('Help', False)
	if Help:
		# print(info)
		print(inspect.getdoc(GetMask))
		return 0

	LowCutoff = kwargs.get('LowCutoff', False)
	HighCutoff = kwargs.get('HighCutoff', False)
	PlotMasks = kwargs.get('PlotMasks', False)
		
	if isinstance(LowCutoff, bool):
		## If no cutoff input, don't mask anything
		mask_low = np.zeros(frame.shape).astype('bool')
	else:
		frame_masqued_low = np.ma.masked_less_equal(frame, LowCutoff)
		mask_low = np.ma.getmaskarray(frame_masqued_low)

	if isinstance(HighCutoff, bool):
		## If no cutoff input, don't mask anything
		mask_high = np.zeros(frame.shape).astype('bool')
	else:
		frame_masqued_high = np.ma.masked_greater_equal(frame, HighCutoff)
		mask_high = np.ma.getmaskarray(frame_masqued_high)

	## Combine low and high cutoff masks. 
	## Make sure that the shape of the array is conserved even if no mask
	mask_combined = np.ma.mask_or(mask_low, mask_high, shrink=False)

	if PlotMasks:
		fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(13,3.5))

		m, M = HySE.UserTools.FindPlottingRange(frame)
		im0 = ax[0].imshow(frame, vmin=m, vmax=M)
		ax[0].set_title('frame')
		divider = make_axes_locatable(ax[0])
		cax = divider.append_axes('right', size='5%', pad=0.05)
		plt.colorbar(im0, cax=cax)

		im1 = ax[1].imshow(frame_masqued_low, vmin=m, vmax=M)
		ax[1].set_title('frame_masqued - Low values')
		divider = make_axes_locatable(ax[1])
		cax = divider.append_axes('right', size='5%', pad=0.05)
		plt.colorbar(im1, cax=cax)

		im2 = ax[2].imshow(frame_masqued_high, vmin=m, vmax=M)
		ax[2].set_title('frame_masqued - High values')
		divider = make_axes_locatable(ax[2])
		cax = divider.append_axes('right', size='5%', pad=0.05)
		plt.colorbar(im2, cax=cax)

		plt.tight_layout()
		plt.show()
	return mask_combined


def GetBestEdgeMask(Masks):
	"""
	Takes in array of shape (Nwav, Y, X) containing a mask for each wavelength
	Outputs a single 2D mask that hides pixels that were masked for any wavelength

	Inputs:
		- Masks

	Ouputs:
		- CombinedMask

	"""
	if Masks.ndim != 3:
		print(f'Input masks must have shape (Nwav, Y, X) -> {Masks.shape}')
	
	CombinedMask = np.any(Masks > 0, axis=0).astype(np.uint8)
	return CombinedMask



