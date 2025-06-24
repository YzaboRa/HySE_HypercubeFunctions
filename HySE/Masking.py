"""

Functions used handle masks

"""


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
import SimpleITK as sitk
import time
# from tqdm.notebook import trange, tqdm, tnrange
from tqdm import trange
# import matplotlib.colors as colors
# import matplotlib.cm as cmx
import inspect
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"


# import HySE_ImportData
import HySE.UserTools
# import HySE.ManipulateHypercube


def GetStandardMask(WhiteCalibration, **kwargs):
	"""
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

# def CombineMasks(mask_white, mask_shifted):
# 	mask = np.ma.mask_or(mask_white, mask_shifted)
# 	mask = mask*1
# 	mask = mask.astype('uint8')
# 	return mask
	

def GetMask(frame, **kwargs):
	'''
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





