"""

Functions that relate to co-registration


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

matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"


import HySE_UserTools
import HySE_ImportData
import HySE_ManipulateHypercube


PythonEnvironment = get_ipython().__class__.__name__


def GetCoregisteredHypercube(vidPath, EdgePos, Nsweep, Wavelengths_list, **kwargs):
	"""
	This function imports the raw data from a single sweep and computes the co-registered
	hypercube from it.
	Inputs:
		- vidPath: where to find the data
		- EdgePos: Positions indicating where each sections of frames is for each wavelength  
			for all sweeps in the dataset  (output from FindHypercube)
		- Nsweep: number of the sweep to look at
		- Wavelnegths_list: list of the wavelengths (unsorted, as done in experiment)
		- kwargs: optional inputs
			- CropImDimensions = [xstart, xend, ystart, yend] : where to crop frames to just keep the image 
				(default values from CCRC HD video)
			- Buffer: sets the numner of frames to ignore on either side of a colour transition
				Totale number of frames removed = 2*Buffer (default 6)
			- ImStatic_Plateau: sets the plateau (wavelength) from which the static image is selected (default 1)
			- ImStatic_Index: sets which frame in the selected plateau (wavelength) as the static image (default 8)
			- PlotDiff: Whether to plot figure showing the co-registration (default False)
				If set to True, also expects:
				- SavingPath: Where to save figure (default '')
				- Plot_PlateauList: for which plateau(x) to plot figure. Aceepts a list of integers or "All" for all plateau (defaul 5)
				- Plot_Index: which frame (index) to plot for each selected plateau (default 14)
			- SaveHypercube: whether or not to save the hypercybe and the sorted wavelengths as npz format
				(default True)

	Output:
	- Hypercube: Sorted hypercube

		Saved:
		if SaveHypercube=True
		- Hypercube (as npz file) for hypercube visualiser
		- Sorted Wavelengths (as npz file) for hypercube visualiser

		if PlotDiff=True
		- plots of the coregistration for wavelengths in Plot_PlateauList and indices=Plot_Index

	"""

	## ImportDatafor the sweep
	DataSweep = HySE_ImportData.GetSweepData_FromPath(vidPath, EdgePos, Nsweep, **kwargs)
	## Compute Hypercube
	Hypercube = SweepCoRegister(DataSweep, Wavelengths_list, **kwargs)
	return Hypercube




def SweepCoRegister_WithNormalisation(DataSweep, WhiteHypercube, Dark, Wavelengths_list, **kwargs):
	"""
	Apply Simple Elastix co-registration to all sweep

	Input:
		- DataSweep: List of 3D arrays. Each element in the list contains all frames in a plateau (wavelength)
		- WhiteHypercube: 
		- kwargs 
			- Buffer: sets the numner of frames to ignore on either side of a colour transition
				Totale number of frames removed = 2*Buffer (default 6)
			- ImStatic_Plateau: sets the plateau (wavelength) from which the static image is selected (default 1)
			- ImStatic_Index: sets which frame in the selected plateau (wavelength) as the static image (default 8)
			- PlotDiff: Whether to plot figure showing the co-registration (default False)
				If set to True, also expects:
				- SavingPath: Where to save figure (default '')
				- Plot_PlateauList: for which plateau(x) to plot figure. Aceepts a list of integers or "All" for all plateau (defaul 5)
				- Plot_Index: which frame (index) to plot for each selected plateau (default 14)
			- SaveHypercube: whether or not to save the hypercybe and the sorted wavelengths as npz format
				(default True)


	Outputs:
		- Normalised and co-registered Hypercube

	"""
	AllIndices = [DataSweep[i].shape[0] for i in range(0,len(DataSweep))]
	MaxIndex = np.amax(AllIndices)
	MinIndex = np.amin(AllIndices)
	# print(AllIndices)
	# print(MaxIndex)

	try:
		Buffer = kwargs['Buffer']
	except KeyError:
		Buffer = 6

	try:
		ImStatic_Plateau = kwargs['ImStatic_Plateau']
		if ImStatic_Plateau==8:
			print(f'Careful! You have set ImStatic_Plateau to 8, which is typically a dark. If this is the case, the co-registration will fail')
	except KeyError:
		ImStatic_Plateau = 1

	try:
		ImStatic_Index = kwargs['ImStatic_Index']
		if ImStatic_Index<5 or ImStatic_Index<Buffer:
			print(f'Careful! You have set ImStatic_Index < 5 or < Buffer ')
			print(f'	This is risks being in the range of unreliable frames too close to a colour transition.')
		if ImStatic_Index>(MinIndex-Buffer):
			print(f'Careful! You have set ImStatic_Index  > (MinIndex - Buffer')
			print(f'	This is risks being in the range of unreliable frames too close to a colour transition.')
	except KeyError:
		ImStatic_Index = 8
		if MinIndex>ImStatic_Index:
			ImStatic_Index = int(MinIndex/2)
			print(f'ImStatic_Index is outside default range. Set to {ImStatic_Index}, please set manually with ImStatic_Index')

	try: 
		PlotDiff = kwargs['PlotDiff']
	except KeyError:
		PlotDiff = False

	if PlotDiff:
		print(f'PlotDiff set to True. Use \'Plot_PlateauList=[]\' or \'All\' and Plot_Index=int to set')
	
	try: 
		SavingPath = kwargs['SavingPath']
	except KeyError:
		SavingPath = ''
		print(f'PlotDiff has been set to True. Indicate a SavingPath.')
	
	try: 
		Plot_PlateauList = kwargs['Plot_PlateauList']
		if isinstance(Plot_PlateauList, int):
			Plot_PlateauList = [Plot_PlateauList]
	except:
		Plot_PlateauList = [5]
	

	try: 
		Plot_Index = kwargs['Plot_Index']
		if Plot_Index<Buffer or Plot_Index>(MinIndex-Buffer):
			print(f'PlotIndex is outside the range of indices that will be analyse ({Buffer}, {MinIndex-Buffer})')
			Plot_Index = int(MinIndex/2)
			print(f'	Seeting it to {PlotIndex}')
	except:
		Plot_Index = 14
		print(f'MinIndex = {MinIndex}, MinIndex-Buffer = {MinIndex-Buffer}')
		if Plot_Index>(MinIndex-Buffer):
			Plot_Index = int(MinIndex/2)
			print(f'Plot_Index outside default range. Set to {Plot_Index}, please set manually with Plot_Index')

	try:
		SaveHypercube = kwargs['SaveHypercube']
		print(f'SaveHypercube set to {SaveHypercube}')
	except:
		SaveHypercube = True

	print(f'Static image: plateau {ImStatic_Plateau}, index {ImStatic_Index}. Use ImStatic_Plateau and ImStatic_Index to change it.')
	print(f'Buffer set to {Buffer}')


	t0 = time.time()
	Ncolours = len(DataSweep)
	(_, YY, XX) = DataSweep[1].shape

	## Deal with special cases when plateau list is input as string
	if isinstance(Plot_PlateauList, str):
		if Plot_PlateauList=='All':
			Plot_PlateauList = [i for i in range(0,Ncolours)]
		elif Plot_PlateauList=='None':
			Plot_PlateauList = []


	## Sort Wavelengths
	order_list = np.argsort(Wavelengths_list)
	Wavelengths_sorted = Wavelengths_list[order_list]

	Hypercube = []

	## Define static image
	im_static = DataSweep[ImStatic_Plateau][ImStatic_Index,:,:]
	im_staticN = HySE_ManipulateHypercube.normalise(im_static, WhiteHypercube[ImStatic_Index,:,:])

	## Loop through all colours (wavelengths)
	print(f'\n Plot_PlateauList = {Plot_PlateauList}, Plot_Index = {Plot_Index}\n')

	for c in range(0, Ncolours):
		if c==8: ## ignore dark
			# print(f'DARK')
			pass
		else:
			ImagesTemp = []
			(NN, YY, XX) = DataSweep[c].shape
			for i in range(Buffer,NN-Buffer):
				im_shifted = DataSweep[c][i,:,:]
				im_white = WhiteHypercube[c,:,:]

				im_shiftedN = HySE_ManipulateHypercube.NormaliseFrames(im_shifted, im_white, Dark)

				im_coregistered, shift_val, time_taken = CoRegisterImages(im_staticN, im_shiftedN)
				ImagesTemp.append(im_coregistered)

				## Plot co-registration is requested
				if PlotDiff:
					# print(f'c={c}, i={i}')
					if c in Plot_PlateauList:
						if '.png' in SavingPath:
							NameTot = SavingPath.split('/')[-1]
							Name = NameTot.replace('.png', '')+f'_Plateau{c}_Index{i}.png'
							SavingPathWithName = SavingPath.replace(NameTot, Name)
						else:
							Name = f'Plateau{c}_Plateau{c}_Index{i}_CoRegistration.png'
							SavingPathWithName = SavingPath+Name

						if i==Plot_Index:
							if c==ImStatic_Plateau and i==ImStatic_Index:
								print(f'Skipping plot for plateau={c}, index={i} because it is the static image')
							else:
								HySE_UserTools.PlotCoRegistered(im_static, im_shifted, im_coregistered, SavePlot=True, SavingPathWithName=SavingPathWithName)

			
			ImagesTemp = np.array(ImagesTemp)
			ImAvg = np.average(ImagesTemp, axis=0)
			Hypercube.append(ImAvg)
			
	tf = time.time()
	Hypercube = np.array(Hypercube)
	## Calculate time taken
	time_total = tf-t0
	minutes = int(time_total/60)
	seconds = time_total - minutes*60
	print(f'\n\n Co-registration took {minutes} min and {seconds:.0f} s in total\n')

	## Sort hypercube according to the order_list
	## Ensures wavelenghts are ordered from blue to red
	Hypercube_sorted = []
	for k in range(0,Hypercube.shape[0]):
		Hypercube_sorted.append(Hypercube[order_list[k]])
	Hypercube_sorted = np.array(Hypercube_sorted)


	if SaveHypercube:
		if '.png' in SavingPath:
			NameTot = SavingPath.split('/')[-1]
			Name = NameTot.replace('.png', '')+f'_CoregisteredHypercube.npz'
			Name_wav = NameTot.replace('.png', '')+f'_CoregisteredHypercube_wavelengths.npz'
			SavingPathHypercube = SavingPath.replace(NameTot, Name)
			SavingPathWavelengths = SavingPath.replace(NameTot, Name_wav)
		else:
			Name = f'_CoregisteredHypercube.npz'
			Name_wav = f'_CoregisteredHypercube_wavelengths.npz'
			SavingPathHypercube = SavingPath+Name
			SavingPathWavelengths = SavingPath+Name_wav

		np.savez(f'{SavingPathHypercube}', Hypercube)
		np.savez(f'{SavingPathWavelengths}', Wavelengths_sorted)

	return Hypercube_sorted









def SweepCoRegister(DataSweep, Wavelengths_list, **kwargs):
	"""
	Apply Simple Elastix co-registration to all sweep

	Input:
		- DataSweep: List of 3D arrays. Each element in the list contains all frames in a plateau (wavelength)
		- kwargs 
			- Buffer: sets the numner of frames to ignore on either side of a colour transition
				Totale number of frames removed = 2*Buffer (default 6)
			- ImStatic_Plateau: sets the plateau (wavelength) from which the static image is selected (default 1)
			- ImStatic_Index: sets which frame in the selected plateau (wavelength) as the static image (default 8)
			- PlotDiff: Whether to plot figure showing the co-registration (default False)
				If set to True, also expects:
				- SavingPath: Where to save figure (default '')
				- Plot_PlateauList: for which plateau(x) to plot figure. Aceepts a list of integers or "All" for all plateau (defaul 5)
				- Plot_Index: which frame (index) to plot for each selected plateau (default 14)
			- SaveHypercube: whether or not to save the hypercybe and the sorted wavelengths as npz format
				(default True)


	Outputs:

	"""
	AllIndices = [DataSweep[i].shape[0] for i in range(0,len(DataSweep))]
	MaxIndex = np.amax(AllIndices)
	MinIndex = np.amin(AllIndices)
	# print(AllIndices)
	# print(MaxIndex)

	try:
		Buffer = kwargs['Buffer']
	except KeyError:
		Buffer = 6

	try:
		ImStatic_Plateau = kwargs['ImStatic_Plateau']
		if ImStatic_Plateau==8:
			print(f'Careful! You have set ImStatic_Plateau to 8, which is typically a dark. If this is the case, the co-registration will fail')
	except KeyError:
		ImStatic_Plateau = 1

	try:
		ImStatic_Index = kwargs['ImStatic_Index']
		if ImStatic_Index<5 or ImStatic_Index<Buffer:
			print(f'Careful! You have set ImStatic_Index < 5 or < Buffer ')
			print(f'	This is risks being in the range of unreliable frames too close to a colour transition.')
		if ImStatic_Index>(MinIndex-Buffer):
			print(f'Careful! You have set ImStatic_Index  > (MinIndex - Buffer')
			print(f'	This is risks being in the range of unreliable frames too close to a colour transition.')
	except KeyError:
		ImStatic_Index = 8
		if MinIndex>ImStatic_Index:
			ImStatic_Index = int(MinIndex/2)
			print(f'ImStatic_Index is outside default range. Set to {ImStatic_Index}, please set manually with ImStatic_Index')

	try: 
		PlotDiff = kwargs['PlotDiff']
	except KeyError:
		PlotDiff = False

	if PlotDiff:
		print(f'PlotDiff set to True. Use \'Plot_PlateauList=[]\' or \'All\' and Plot_Index=int to set')
	
	try: 
		SavingPath = kwargs['SavingPath']
	except KeyError:
		SavingPath = ''
		print(f'PlotDiff has been set to True. Indicate a SavingPath.')
	
	try: 
		Plot_PlateauList = kwargs['Plot_PlateauList']
		if isinstance(Plot_PlateauList, int):
			Plot_PlateauList = [Plot_PlateauList]
	except:
		Plot_PlateauList = [5]
	

	try: 
		Plot_Index = kwargs['Plot_Index']
		if Plot_Index<Buffer or Plot_Index>(MinIndex-Buffer):
			print(f'PlotIndex is outside the range of indices that will be analyse ({Buffer}, {MinIndex-Buffer})')
			Plot_Index = int(MinIndex/2)
			print(f'	Seeting it to {PlotIndex}')
	except:
		Plot_Index = 14
		print(f'MinIndex = {MinIndex}, MinIndex-Buffer = {MinIndex-Buffer}')
		if Plot_Index>(MinIndex-Buffer):
			Plot_Index = int(MinIndex/2)
			print(f'Plot_Index outside default range. Set to {Plot_Index}, please set manually with Plot_Index')

	try:
		SaveHypercube = kwargs['SaveHypercube']
		print(f'SaveHypercube set to {SaveHypercube}')
	except:
		SaveHypercube = True

	print(f'Static image: plateau {ImStatic_Plateau}, index {ImStatic_Index}. Use ImStatic_Plateau and ImStatic_Index to change it.')
	print(f'Buffer set to {Buffer}')


	t0 = time.time()
	Ncolours = len(DataSweep)
	(_, YY, XX) = DataSweep[1].shape

	## Deal with special cases when plateau list is input as string
	if isinstance(Plot_PlateauList, str):
		if Plot_PlateauList=='All':
			Plot_PlateauList = [i for i in range(0,Ncolours)]
		elif Plot_PlateauList=='None':
			Plot_PlateauList = []


	## Sort Wavelengths
	order_list = np.argsort(Wavelengths_list)
	Wavelengths_sorted = Wavelengths_list[order_list]

	Hypercube = []

	## Define static image
	im_static = DataSweep[ImStatic_Plateau][ImStatic_Index,:,:]

	## Loop through all colours (wavelengths)
	print(f'\n Plot_PlateauList = {Plot_PlateauList}, Plot_Index = {Plot_Index}\n')

	for c in range(0, Ncolours):
		if c==8: ## ignore dark
			# print(f'DARK')
			pass
		else:
			ImagesTemp = []
			(NN, YY, XX) = DataSweep[c].shape
			for i in range(Buffer,NN-Buffer):
				im_shifted = DataSweep[c][i,:,:]
				im_coregistered, shift_val, time_taken = CoRegisterImages(im_static, im_shifted)
				ImagesTemp.append(im_coregistered)

				## Plot co-registration is requested
				if PlotDiff:
					# print(f'c={c}, i={i}')
					if c in Plot_PlateauList:
						if '.png' in SavingPath:
							NameTot = SavingPath.split('/')[-1]
							Name = NameTot.replace('.png', '')+f'_Plateau{c}_Index{i}.png'
							SavingPathWithName = SavingPath.replace(NameTot, Name)
						else:
							Name = f'Plateau{c}_Plateau{c}_Index{i}_CoRegistration.png'
							SavingPathWithName = SavingPath+Name

						if i==Plot_Index:
							if c==ImStatic_Plateau and i==ImStatic_Index:
								print(f'Skipping plot for plateau={c}, index={i} because it is the static image')
							else:
								HySE_UserTools.PlotCoRegistered(im_static, im_shifted, im_coregistered, SavePlot=True, SavingPathWithName=SavingPathWithName)

			
			ImagesTemp = np.array(ImagesTemp)
			ImAvg = np.average(ImagesTemp, axis=0)
			Hypercube.append(ImAvg)
			
	tf = time.time()
	Hypercube = np.array(Hypercube)
	## Calculate time taken
	time_total = tf-t0
	minutes = int(time_total/60)
	seconds = time_total - minutes*60
	print(f'\n\n Co-registration took {minutes} min and {seconds:.0f} s in total\n')

	## Sort hypercube according to the order_list
	## Ensures wavelenghts are ordered from blue to red
	Hypercube_sorted = []
	for k in range(0,Hypercube.shape[0]):
		Hypercube_sorted.append(Hypercube[order_list[k]])
	Hypercube_sorted = np.array(Hypercube_sorted)


	if SaveHypercube:
		if '.png' in SavingPath:
			NameTot = SavingPath.split('/')[-1]
			Name = NameTot.replace('.png', '')+f'_CoregisteredHypercube.npz'
			Name_wav = NameTot.replace('.png', '')+f'_CoregisteredHypercube_wavelengths.npz'
			SavingPathHypercube = SavingPath.replace(NameTot, Name)
			SavingPathWavelengths = SavingPath.replace(NameTot, Name_wav)
		else:
			Name = f'_CoregisteredHypercube.npz'
			Name_wav = f'_CoregisteredHypercube_wavelengths.npz'
			SavingPathHypercube = SavingPath+Name
			SavingPathWavelengths = SavingPath+Name_wav

		np.savez(f'{SavingPathHypercube}', Hypercube)
		np.savez(f'{SavingPathWavelengths}', Wavelengths_sorted)

	return Hypercube_sorted





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



