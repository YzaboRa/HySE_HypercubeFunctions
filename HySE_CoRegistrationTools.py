"""



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
from tqdm.notebook import trange, tqdm, tnrange

matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"



def GetSweepData_FromPath(vidPath, EdgePos, Nsweep, **kwargs):
	## Check if the user has specificed the image crop dimensions
	try:
		CropImDimensions = kwargs['CropImDimensions']
		CropImDimensions_input = True
	except KeyError:
		CropImDimensions_input = False

	## Import all the data
	if CropImDimensions_input:
		DataAll = ImportData(vidPath, CropImDimensions=CropImDimensions)
	else:
		DataAll = ImportData(vidPath)

	DataSweep = []
	for Nc in range(0,len(EdgePos[Nsweep])):
		Data_c = DataAll[EdgePos[Nsweep][Nc,0]:EdgePos[Nsweep][Nc,0]+EdgePos[Nsweep][Nc,1], :,:]
		DataSweep.append(Data_c)

	return DataSweep




def CoRegisterImages(im_static, im_shifted):
	t0 = time.time()
	## Convert the numpy array to simple elestix format
	im_static_se = sitk.GetImageFromArray(im_static)
	im_shifted_se = sitk.GetImageFromArray(im_shifted)

	## Create object
	elastixImageFilter = sitk.ElastixImageFilter()
	
	## Turn off console
	elastixImageFilter.LogToConsoleOff()
	
	## Set image parameters
	elastixImageFilter.SetFixedImage(im_static_se)
	elastixImageFilter.SetMovingImage(im_shifted_se)
	
	## Set transform parameters
	parameterMap = sitk.GetDefaultParameterMap('translation')
	parameterMap['Transform'] = ['BSplineTransform']
	
	## Parameters to play with if co-registration is not optimal:
	
#         # Controls how long the optimizer runs
#     parameterMap['MaximumNumberOfIterations'] = ['500'] 
#         # You can try different metrics like AdvancedMattesMutualInformation, NormalizedCorrelation, 
#         # or AdvancedKappaStatistic for different registration scenarios.
#     parameterMap['Metric'] = ['AdvancedMattesMutualInformation']
#         # Adjust the number of bins used in mutual information metrics
#     parameterMap['NumberOfHistogramBins'] = ['32']
#         # Change the optimizer to AdaptiveStochasticGradientDescent for potentially better convergence
#     parameterMap['Optimizer'] = ['AdaptiveStochasticGradientDescent']
#         # Controls the grid spacing for the BSpline transform
#     parameterMap['FinalGridSpacingInPhysicalUnits'] = ['10.0']
#         # Refines the BSpline grid at different resolutions.
#     parameterMap['GridSpacingSchedule'] = ['10.0', '5.0', '2.0']
#         # Automatically estimate the scales for the transform parameters.
#     parameterMap['AutomaticScalesEstimation'] = ['true']
#         # Controls the number of resolutions used in the multi-resolution pyramid. 
#         # A higher number can lead to better registration at the cost of increased computation time.
#     parameterMap['NumberOfResolutions'] = ['4']
#         # Automatically initializes the transform based on the center of mass of the images.
#     parameterMap['AutomaticTransformInitialization'] = ['true']
#         # Controls the interpolation order for the final transformation.
#     parameterMap['FinalBSplineInterpolationOrder'] = ['3']
	
# #         # Adjust the maximum step length for the optimizer
# #     parameterMap['MaximumStepLength'] = ['4.0']
# #         # Use more samples for computing gradients
# #     parameterMap['NumberOfSamplesForExactGradient'] = ['10000']
# #         # Specify the grid spacing in voxels for the final resolution.
# #     parameterMap['FinalGridSpacingInVoxels'] = ['8.0']
# #         # Defines the spacing of the sampling grid used during optimization.
# #     parameterMap['SampleGridSpacing'] = ['2.0']
		
	
	## If required, set maximum number of iterations
#     parameterMap['MaximumNumberOfIterations'] = ['500']
	elastixImageFilter.SetParameterMap(parameterMap)
	
	## Execute
	result = elastixImageFilter.Execute()
	## Convert result to numpy array
	im_coregistered = sitk.GetArrayFromImage(result)
	t1 = time.time()
	
	## Find time taken:
	time_taken = t1-t0
	
	## Get an idea of difference
	shift_val = np.average(np.abs(np.subtract(im_static,im_coregistered)))
	
	## return 
	return im_coregistered, shift_val, time_taken



	