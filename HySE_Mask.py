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
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"


# import HySE_ImportData
import HySE_UserTools

def ConvertMaskToBinary(mask):
	mask_binary = 1-mask*1
	return mask_binary.astype('uint8')
	

def GetMask(frame, **kwargs):
	info='''
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
	try:
		Help = kwargs['Help']
	except KeyError:
		Help = False
	if Help:
		print(info)
		return 0
		
	try:
		LowCutoff = kwargs['LowCutoff']
	except KeyError:
		LowCutoff = 0.8
		
	try:
		HighCutoff = kwargs['HighCutoff']
	except KeyError:
		HighCutoff = np.nanmax(frame)+1
	
	try:
		PlotMasks = kwargs['PlotMasks']
	except KeyError:
		PlotMasks = False
		
	frame_masqued_low = np.ma.masked_less_equal(frame, LowCutoff)
	mask_low = np.ma.getmaskarray(frame_masqued_low)
	
	frame_masqued_high = np.ma.masked_greater_equal(frame, HighCutoff)
	mask_high = np.ma.getmaskarray(frame_masqued_high)
	
	mask_combined = np.ma.mask_or(mask_low, mask_high)

	if PlotMasks:
		fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(13,3.5))

		m, M = HySE_UserTools.FindPlottingRange(frame)
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



def CoRegisterImages_WithMask(im_static, im_moving, **kwargs):
	info='''
	Function to co-register two images. Allows the option to mask some regions of both images.
	Input:
		- im_static: 2D numpy array
		- im_moving: 23 numpy array, same size as im_static
		- kwargs:
			- StaticMask: 2d numpy array, same size as im_static. Type uint8 or bool_
			- MovingMask: 2d numpy array, same size as im_static. Type uint8 or bool_
			- Affine: whether to apply affine transform instead of Bspline (default False)
			- Verbose: wheter to enable the console output from elastix (default False)
				NB: signficant output. Do no enable executing in a loop
			- Help
	Output:
		- im_coregistered
	
	'''
	try: 
		Help = kwargs['Help']
	except KeyError:
		Help = False
	if Help:
		print(info)
		return 0
		
	try: 
		Affine = kwargs['Affine']
	except KeyError:
		Affine = False

	try: 
		Verbose = kwargs['Verbose']
	except KeyError:
		Verbose = False
		
	## Convert the numpy array to simple elestix format
	im_static_se = sitk.GetImageFromArray(im_static)
	im_moving_se = sitk.GetImageFromArray(im_moving)
	
	## Create object
	elastixImageFilter = sitk.ElastixImageFilter()

	## Turn off console
	if Verbose==False:
		elastixImageFilter.LogToConsoleOff()

	## Set image parameters
	elastixImageFilter.SetFixedImage(im_static_se)
	elastixImageFilter.SetMovingImage(im_moving_se)

	## Check if user has set a mask for the stating image
	try: 
		StaticMask = kwargs['StaticMask']
		if type(StaticMask[0,0])==np.bool_:
			print(f'Boolean mask. Converting to binary')
			StaticMask = ConvertMaskToBinary(StaticMask)
		elif type(StaticMask[0,0])!=np.uint8:
			print(f'StaticMask is neither in uint8 or boolean format, code won\'t run')
		StaticMask_se = sitk.GetImageFromArray(StaticMask)
		elastixImageFilter.SetFixedMask(StaticMask_se)
	except KeyError:
		pass
	
	## Check if user has set a mask for the moving image
	try: 
		MovingMask = kwargs['MovingMask']
		if type(MovingMask[0,0])==np.bool_:
			print(f'Boolean mask. Converting to binary')
			MovingMask = ConvertMaskToBinary(MovingMask)
		elif type(MovingMask[0,0])!=np.uint8:
			print(f'MovingMask is neither in uint8 or boolean format, code won\'t run')
		MovingMask_se = sitk.GetImageFromArray(MovingMask)
		elastixImageFilter.SetFixedMask(MovingMask)
	except KeyError:
		pass
	

	## Set transform parameters
	if Affine:
		parameterMap = sitk.GetDefaultParameterMap('affine')
	else:
		parameterMap = sitk.GetDefaultParameterMap('translation')
		## Select metric robust to intensity differences (non uniform)
		parameterMap['Metric'] = ['AdvancedMattesMutualInformation'] 
		## Select Bspline transform, which allows for non rigid and non uniform deformations
		parameterMap['Transform'] = ['BSplineTransform']
		
#     parameterMap["UseFixedMask"] = ["true"]
#     parameterMap["UseMovingMask"] = ["true"]

	elastixImageFilter.SetParameterMap(parameterMap)

	## Execute
	result = elastixImageFilter.Execute()
	## Convert result to numpy array
	im_coregistered = sitk.GetArrayFromImage(result)
	return im_coregistered
