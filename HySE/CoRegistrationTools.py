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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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




def GetCoregisteredHypercube(vidPath, EdgePos, Nsweep, Wavelengths_list, **kwargs):
	"""

	~~ Co-registration only ~~

	This function imports the raw data from a single sweep and computes the co-registered
	hypercube from it.
	Uses GetSweepData_FromPath() and SweepCoRegister() functions (which uses CoRegisterImages() for the registration)

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

		Used in SweepCoregister/CoRegisterImages:
			- 

	Output:
	- Hypercube: Sorted hypercube

		Saved:
		if SaveHypercube=True
		- Hypercube (as npz file) for hypercube visualiser
		- Sorted Wavelengths (as npz file) for hypercube visualiser

		if PlotDiff=True
		- plots of the coregistration for wavelengths in Plot_PlateauList and indices=Plot_Index

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetCoregisteredHypercube))
		return 0

	## ImportDatafor the sweep
	DataSweep = HySE.Import.GetSweepData_FromPath(vidPath, EdgePos, Nsweep, **kwargs)
	## Compute Hypercube
	Hypercube, AllTransforms = SweepCoRegister(DataSweep, Wavelengths_list, **kwargs)
	return Hypercube, AllTransforms




# def SweepCoRegister_WithNormalisation(DataSweep, WhiteHypercube, Dark, Wavelengths_list, **kwargs):
# 	"""

# 	~~ Co-registration with normalisation and masking ~~

# 	NB: Old function, normalisation and masking not optimal. Never tested with wavelength mixing.
# 		New function to be written.

# 	Apply Simple Elastix co-registration to all sweep

# 	Input:
# 		- DataSweep: List of 3D arrays. Each element in the list contains all frames in a plateau (wavelength)
# 		- WhiteHypercube: 
# 		- kwargs 
# 			- Buffer: sets the numner of frames to ignore on either side of a colour transition
# 				Totale number of frames removed = 2*Buffer (default 6)
# 			- JustBright = False : If only considering the brightest frames (of the RGB frames)
# 			- ImStatic_Plateau: sets the plateau (wavelength) from which the static image is selected (default 1)
# 			- ImStatic_Index: sets which frame in the selected plateau (wavelength) as the static image (default 8)
# 			- PlotDiff: Whether to plot figure showing the co-registration (default False)
# 				If set to True, also expects:
# 				- SavingPath: Where to save figure (default '')
# 				- Plot_PlateauList: for which plateau(x) to plot figure. Aceepts a list of integers or "All" for all plateau (defaul 5)
# 				- Plot_Index: which frame (index) to plot for each selected plateau (default 14)
# 			- SaveHypercube: whether or not to save the hypercybe and the sorted wavelengths as npz format
# 				(default True)
# 			- BrightFramesOnly = False : If True, only averages bright frames


# 	Outputs:
# 		- Normalised and co-registered Hypercube

# 	"""
# 	Help = kwargs.get('Help', False)
# 	if Help:
# 		# print(info)
# 		print(inspect.getdoc(SweepCoRegister_WithNormalisation))
# 		return 0

# 	AllIndices = [DataSweep[i].shape[0] for i in range(0,len(DataSweep))]
# 	MaxIndex = np.amax(AllIndices)
# 	MinIndex = np.amin(AllIndices)
# 	# print(AllIndices)
# 	# print(MaxIndex)

# 	Buffer = kwargs.get('Buffer', 6)

# 	ImStatic_Plateau = kwargs.get('ImStatic_Plateau', 1)
# 	if ImStatic_Plateau==8:
# 		print(f'Careful! You have set ImStatic_Plateau to 8, which is typically a dark. If this is the case, the co-registration will fail')


# 	ImStatic_Index = kwargs.get('ImStatic_Index')
# 	if ImStatic_Index is None:
# 		ImStatic_Index = 8
# 		if MinIndex>ImStatic_Index:
# 			ImStatic_Index = int(MinIndex/2)
# 			print(f'ImStatic_Index is outside default range. Set to {ImStatic_Index}, please set manually with ImStatic_Index')
# 	else:
# 		if ImStatic_Index<5 or ImStatic_Index<Buffer:
# 			print(f'Careful! You have set ImStatic_Index < 5 or < Buffer ')
# 			print(f'	This is risks being in the range of unreliable frames too close to a colour transition.')
# 		if ImStatic_Index>(MinIndex-Buffer):
# 			print(f'Careful! You have set ImStatic_Index  > (MinIndex - Buffer')
# 			print(f'	This is risks being in the range of unreliable frames too close to a colour transition.')

# 	PlotDiff = kwargs.get('PlotDiff', False)
# 	if PlotDiff:
# 		print(f'PlotDiff set to True. Use \'Plot_PlateauList=[]\' or \'All\' and Plot_Index=int to set')
	


# 	SavingPath = kwargs.get('SavingPath')
# 	if SavingPath is None:
# 		print(f'PlotDiff has been set to True. Indicate a SavingPath.')
	

# 	Plot_PlateauList = kwargs.get('Plot_PlateauList')
# 	if Plot_PlateauList is None:
# 		Plot_PlateauList = [5]
# 		print(f'Set Plot_PlateauList and Plot_Index to set images to plot')
# 	else:
# 		if isinstance(Plot_PlateauList, int):
# 			Plot_PlateauList = [Plot_PlateauList]

	

# 	Plot_Index = kwargs.get('Plot_Index')
# 	if Plot_Index is None:
# 		Plot_Index = 14
# 		print(f'MinIndex = {MinIndex}, MinIndex-Buffer = {MinIndex-Buffer}')
# 		if Plot_Index>(MinIndex-Buffer):
# 			Plot_Index = int(MinIndex/2)
# 			print(f'Plot_Index outside default range. Set to {Plot_Index}, please set manually with Plot_Index')
# 	else:
# 		if Plot_Index<Buffer or Plot_Index>(MinIndex-Buffer):
# 			print(f'PlotIndex is outside the range of indices that will be analyse ({Buffer}, {MinIndex-Buffer})')
# 			Plot_Index = int(MinIndex/2)
# 			print(f'	Seeting it to {PlotIndex}')



# 	SaveHypercube = kwargs.get('SaveHypercube', True)
# 	if SaveHypercube:
# 		print(f'Saving Hypercube')


# 	print(f'Static image: plateau {ImStatic_Plateau}, index {ImStatic_Index}. Use ImStatic_Plateau and ImStatic_Index to change it.')
# 	print(f'Buffer set to {Buffer}')


# 	t0 = time.time()
# 	Ncolours = len(DataSweep)
# 	(_, YY, XX) = DataSweep[1].shape

# 	## Deal with special cases when plateau list is input as string
# 	if isinstance(Plot_PlateauList, str):
# 		if Plot_PlateauList=='All':
# 			Plot_PlateauList = [i for i in range(0,Ncolours)]
# 		elif Plot_PlateauList=='None':
# 			Plot_PlateauList = []

# 	JustBright = kwargs.get('JustBright', False)
# 	if JustBright:
# 		print(f'Only averaging bright frames')

# 	## Sort Wavelengths
# 	order_list = np.argsort(Wavelengths_list)
# 	Wavelengths_sorted = Wavelengths_list[order_list]

# 	Hypercube = []

# 	## Define static image
# 	im_static = DataSweep[ImStatic_Plateau][ImStatic_Index,:,:]
# 	ImStatic_Plateau_sorted = order_list[ImStatic_Plateau]
# 	im_staticN = HySE.ManipulateHypercube.NormaliseFrames(im_static, WhiteHypercube[ImStatic_Plateau_sorted,:,:], Dark)

# 	## Loop through all colours (wavelengths)
# 	print(f'\n Plot_PlateauList = {Plot_PlateauList}, Plot_Index = {Plot_Index}\n')

# 	for c in range(0, Ncolours):
# 		if c==8: ## ignore dark
# 			# print(f'DARK')
# 			pass
# 		else:

# 			print(f'Working on: {c} /{Ncolours}')
# 			# ImagesTemp = []
# 			# (NN, YY, XX) = DataSweep[c].shape

# 			## Which of the first 3 images if brightest?
# 			vals = [np.average(DataSweep[c][Buffer+q,:,:]) for q in range(Buffer,Buffer+3)]
			
# 			offset = np.where(vals==np.amax(vals))[0][0]


# 			ImagesTemp = []
# 			(NN, YY, XX) = DataSweep[c].shape
# 			# ## Averaging All frames
# 			# for i in range(Buffer,NN-Buffer):
# 			## Averaing brightest frames
# 			for i in range(Buffer+offset,NN-Buffer,3):
# 				im_shifted = DataSweep[c][i,:,:]
# 				if c>=8: ## the hypercube does not include dark in the middle
# 					hypercube_index = order_list[c-1] ## hypercube is already sorted
# 					im_white = WhiteHypercube[hypercube_index,:,:]
# 				else:
# 					hypercube_index = order_list[c]
# 					im_white = WhiteHypercube[hypercube_index,:,:]

# 				im_shiftedN = HySE.ManipulateHypercube.NormaliseFrames(im_shifted, im_white, Dark)

# 				im_coregistered, shift_val, time_taken = CoRegisterImages(im_staticN, im_shiftedN)
# 				ImagesTemp.append(im_coregistered)

# 				## Plot co-registration is requested
# 				if PlotDiff:
# 					# print(f'c={c}, i={i}')
# 					if c in Plot_PlateauList:
# 						if '.png' in SavingPath:
# 							NameTot = SavingPath.split('/')[-1]
# 							Name = NameTot.replace('.png', '')+f'_Plateau{c}_Index{i}.png'
# 							SavingPathWithName = SavingPath.replace(NameTot, Name)
# 						else:
# 							Name = f'Plateau{c}_Plateau{c}_Index{i}_CoRegistration.png'
# 							SavingPathWithName = SavingPath+Name

# 						if i==Plot_Index:
# 							if c==ImStatic_Plateau and i==ImStatic_Index:
# 								print(f'Skipping plot for plateau={c}, index={i} because it is the static image')
# 							else:
# 								HySE.UserTools.PlotCoRegistered(im_static, im_shifted, im_coregistered, SavePlot=True, SavingPathWithName=SavingPathWithName)
# 								print(f'Saved CoRegistration plot for plateau {c} and index {i} here : {SavingPathWithName}')

			
# 			ImagesTemp = np.array(ImagesTemp)
# 			ImAvg = np.average(ImagesTemp, axis=0)
# 			Hypercube.append(ImAvg)
			
# 	tf = time.time()
# 	Hypercube = np.array(Hypercube)
# 	## Calculate time taken
# 	time_total = tf-t0
# 	minutes = int(time_total/60)
# 	seconds = time_total - minutes*60
# 	print(f'\n\n Co-registration took {minutes} min and {seconds:.0f} s in total\n')

# 	## Sort hypercube according to the order_list
# 	## Ensures wavelenghts are ordered from blue to red
# 	Hypercube_sorted = []
# 	for k in range(0,Hypercube.shape[0]):
# 		Hypercube_sorted.append(Hypercube[order_list[k]])
# 	Hypercube_sorted = np.array(Hypercube_sorted)


# 	if SaveHypercube:
# 		if '.png' in SavingPath:
# 			NameTot = SavingPath.split('/')[-1]
# 			Name = NameTot.replace('.png', '')+f'_CoregisteredHypercube.npz'
# 			Name_wav = NameTot.replace('.png', '')+f'_CoregisteredHypercube_wavelengths.npz'
# 			SavingPathHypercube = SavingPath.replace(NameTot, Name)
# 			SavingPathWavelengths = SavingPath.replace(NameTot, Name_wav)
# 		else:
# 			Name = f'_CoregisteredHypercube.npz'
# 			Name_wav = f'_CoregisteredHypercube_wavelengths.npz'
# 			SavingPathHypercube = SavingPath+Name
# 			SavingPathWavelengths = SavingPath+Name_wav

# 		np.savez(f'{SavingPathHypercube}', Hypercube)
# 		np.savez(f'{SavingPathWavelengths}', Wavelengths_sorted)

# 	return Hypercube_sorted



def SweepCoRegister(DataSweep, Wavelengths_list, **kwargs):
	"""

	~~ Used in SweepCoRegister_WithNormalisation ~~

	NB: Old function, normalisation and masking not optimal. Never tested with wavelength mixing.
		New function to be written.

	Apply Simple Elastix co-registration to all sweep with CoRegisterImages() function

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
			- Frames = [int, int, ...] : If provided, indicates which frames from a single plateau to keep. 
				Accepted values are 0, 1, ..., where frame 0 refers to the first frame that would normally be considered (after removing buffers)
			- EdgeMask : Mask of size (Y,X) that masks the edges. Note that values of 1=masled pixel
			- ReflectionsMasks : Array of masks of size (Nwav, Y, X) indicating pixels where specular reflections might mess up with the registration

	Outputs:
		- Hypercube_sorted

	"""

	Help = kwargs.get('Help', False)
	if Help:
		# print(info)
		print(inspect.getdoc(SweepCoRegister))
		return 0


	AllIndices = [DataSweep[i].shape[0] for i in range(0,len(DataSweep))]
	MaxIndex = np.amax(AllIndices)
	MinIndex = np.amin(AllIndices)
	# print(AllIndices)
	# print(MaxIndex)

	Buffer = kwargs.get('Buffer', 6)
	Frames = kwargs.get('Frames')
	if Frames is not None:
		print(f'Only Coregistring these Frames: {Frames}')

	EdgeMask = kwargs.get('EdgeMask')
	ReflectionsMasks = kwargs.get('ReflectionsMasks')


	ImStatic_Plateau = kwargs.get('ImStatic_Plateau', 1)
	if ImStatic_Plateau==8:
		print(f'Careful! You have set ImStatic_Plateau to 8, which is typically a dark. If this is the case, the co-registration will fail')


	ImStatic_Index = kwargs.get('ImStatic_Index')
	if ImStatic_Index is None:
		ImStatic_Index = 8
		if MinIndex>ImStatic_Index:
			ImStatic_Index = int(MinIndex/2)
			print(f'ImStatic_Index is outside default range. Set to {ImStatic_Index}, please set manually with ImStatic_Index')
	else:
		if ImStatic_Index<5 or ImStatic_Index<Buffer:
			print(f'Careful! You have set ImStatic_Index < 5 or < Buffer ')
			print(f'	This is risks being in the range of unreliable frames too close to a colour transition.')
		if ImStatic_Index>(MinIndex-Buffer):
			print(f'Careful! You have set ImStatic_Index  > (MinIndex - Buffer')
			print(f'	This is risks being in the range of unreliable frames too close to a colour transition.')


	PlotDiff = kwargs.get('PlotDiff', False)
	if PlotDiff:
		print(f'PlotDiff set to True. Use \'Plot_PlateauList=[]\' or \'All\' and Plot_Index=int to set')
	

	SavingPath = kwargs.get('SavingPath')
	if SavingPath is None:
		SavingPath = ''
		print(f'PlotDiff has been set to True. Indicate a SavingPath.')

	
	# BrightFramesOnly = kwargs.get('BrightFramesOnly', False)

	Plot_PlateauList = kwargs.get('Plot_PlateauList', [5])
	if isinstance(Plot_PlateauList, int):
			Plot_PlateauList = [Plot_PlateauList]
	

	Plot_Index = kwargs.get('Plot_Index')
	if Plot_Index is None:
		Plot_Index = 14
		print(f'MinIndex = {MinIndex}, MinIndex-Buffer = {MinIndex-Buffer}')
		if Plot_Index>(MinIndex-Buffer):
			Plot_Index = int(MinIndex/2)
			print(f'Plot_Index outside default range. Set to {Plot_Index}, please set manually with Plot_Index')
	else:
		if Plot_Index<Buffer or Plot_Index>(MinIndex-Buffer):
			print(f'PlotIndex is outside the range of indices that will be analyse ({Buffer}, {MinIndex-Buffer})')
			Plot_Index = int(MinIndex/2)
			print(f'	Seeting it to {Plot_Index}')


	SaveHypercube = kwargs.get('SaveHypercube', True)
	if SaveHypercube:
		print(f'Saving Hypercube')

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
	AllTransforms = []

	## Define static image
	im_static = DataSweep[ImStatic_Plateau][ImStatic_Index,:,:]

	## Loop through all colours (wavelengths)
	print(f'\n Plot_PlateauList = {Plot_PlateauList}, Plot_Index = {Plot_Index}\n')

	for c in range(0, Ncolours):
		if c==8: ## ignore dark
			# print(f'DARK')
			pass
		else:
			print(f'Working on: {c} /{Ncolours}')
			ImagesTemp = []
			(NN, YY, XX) = DataSweep[c].shape

			## Which of the first 3 images if brightest?
			vals = [np.average(DataSweep[c][Buffer+q,:,:]) for q in range(Buffer,Buffer+3)]
			
			offset = np.where(vals==np.amax(vals))[0][0]


			loop_start = Buffer
			loop_end = NN-Buffer
			loop_step = 1
			if Frames is None:
				Nframes_tot = loop_end-loop_start
				Frames = [i for i in range(0,Nframes_tot)]
			for i in range(loop_start, loop_end, loop_step):
				if i-loop_start in Frames:
					# print(f'CoRegistering frame {i-loop_start} in {Frames}')

					im_shifted = DataSweep[c][i,:,:]
					if EdgeMask is not None:
						if ReflectionsMasks is not None:
							im_coregistered, coregister_transform = CoRegisterImages(im_static, im_shifted, ReflectionsMasks=ReflectionsMasks[c,:,:], EdgeMask=EdgeMask, **kwargs) #, **kwargs
						else:
							im_coregistered, coregister_transform = CoRegisterImages(im_static, im_shifted, EdgeMask=EdgeMask, **kwargs) #, **kwargs
					elif ReflectionsMasks is not None:
						im_coregistered, coregister_transform = CoRegisterImages(im_static, im_shifted, ReflectionsMasks=ReflectionsMasks[c,:,:], **kwargs)
					else:
						im_coregistered, coregister_transform = CoRegisterImages(im_static, im_shifted, **kwargs) #, **kwargs
					ImagesTemp.append(im_coregistered)
					AllTransforms.append(coregister_transform)

					if c<8:
						# print(f'      c={c}, i={i}: {Wavelengths_list[c]} nm, avg {np.average(im_shifted)}, shift val {shift_val:.2f}, {time_taken:.2f} s')
						print(f'      c={c}, i={i}: {Wavelengths_list[c]} nm, avg {np.average(im_shifted)}')
					else:
						# print(f'      c={c}, i={i}: {Wavelengths_list[c-1]} nm, avg {np.average(im_shifted)}, shift val {shift_val:.2f}, {time_taken:.2f} s')
						print(f'      c={c}, i={i}: {Wavelengths_list[c-1]} nm, avg {np.average(im_shifted)}')

					## Plot co-registration is requested
					if PlotDiff:
						# print(f'c={c}, i={i}')
						if c in Plot_PlateauList:
							if '.png' in SavingPath:
								NameTot = SavingPath.split('/')[-1]
								Name = NameTot.replace('.png', '')+f'_Plateau{c}_Index{i}.png'
								SavingPathWithName = SavingPath.replace(NameTot, Name)
							else:
								Name = f'_Plateau{c}_Index{i}_CoRegistration.png'
								SavingPathWithName = SavingPath+Name

							if i==Plot_Index:
								if c==ImStatic_Plateau and i==ImStatic_Index:
									print(f'Skipping plot for plateau={c}, index={i} because it is the static image')
								else:
									HySE.UserTools.PlotCoRegistered(im_static, im_shifted, im_coregistered, SavePlot=True, SavingPathWithName=SavingPathWithName)
				# else:
				# 	print(f'Ignoring frame {i-loop_start}, not in {Frames}')

			
			ImagesTemp = np.array(ImagesTemp)

			print(f'  Averaging {len(ImagesTemp)} frames')
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
	AllTransforms_sorted = []
	for k in range(0,Hypercube.shape[0]):
		Hypercube_sorted.append(Hypercube[order_list[k]])
		AllTransforms_sorted.append(AllTransforms[order_list[k]])
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

	return Hypercube_sorted, AllTransforms_sorted




def CoRegisterImages(im_static, im_shifted, **kwargs):
	"""
	Co-registers a shifted image to a defined static image using SimpleElastix (v3)

	Inputs:
		- im_static
		- im_shifted
		- kwargs
			- Affine = False : use affine transform only (True/False)
			- TwoStage = False : perform an affine registration, then a non-rigid (BSpline) registration (True/False).
			- Verbose = False : print logs (True/False)
			- MaximumNumberOfIterations: int
			- Metric = 'AdvancedMattesMutualInformation' : e.g. 'AdvancedMattesMutualInformation', 'AdvancedMeanSquares', 'AdvancedNormalizedCorrelation', 
				'AdvancedKappaStatistic', '', 'NormalizedMutualInformation', 'DisplacementMagnitudePenalty'
			- Optimizer = 'AdaptiveStochasticGradientDescent' : e.g. 'AdaptiveStochasticGradientDescent', 'RegularStepGradientDescent', 'CMAES'
			- Transform = 'BSplineTransform' : e.g. 'BSplineTransform', 'Euler2DTransform', 'Similarity2DTransform', 'AffineTransform', 'TranslationTransform'
			- GradientMagnitude = False: if True, register based on gradient images
			- HistogramMatch = False: if True, match moving image histogram to fixed
			- IntensityNorm = False: if True, z-score normalize both images
			- Blurring = False: if True, apply Gaussian blur
			- Sigma = 2: If blurring images, blur by sigma (Gaussian)
			- EdgeMask: binary mask to exclude non-informative areas
			- ReflectionsMasks: binary mask to exclude areas with specular reflections (saturating) 
				N.B. Both EdgeMask and ReflectionsMask, if specified, are combined and inverted to be fed to the algorithm
			- GridSpacing : Spacing of the B-spline control point grid. A larger value produces a stiffer, smoother transform and reduces artifacts.

	Outputs:
		- Registered Image
		- transform parameter map (to be applied to other data)
			To be used:
				transformixImageFilter = sitk.TransformixImageFilter()
				transformixImageFilter.SetMovingImage(sitk.ReadImage("other_image.tif", sitk.sitkFloat32))
				transformixImageFilter.SetTransformParameterMap(sitk.ReadParameterFile("TransformParameters.0.0.txt"))
				transformixImageFilter.Execute()

	"""
	if _sitk is None:
		raise ImportError("SimpleITK is required. Install it with `pip install SimpleITK`.")

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(CoRegisterImages))
		return 0, 0, 0

	Affine = kwargs.get('Affine', False)
	TwoStage = kwargs.get('TwoStage', False)
	Verbose = kwargs.get('Verbose', False)
	Metric = kwargs.get('Metric', 'AdvancedMattesMutualInformation')
	Transform = kwargs.get('Transform', 'BSplineTransform')
	Optimizer = kwargs.get('Optimizer', 'AdaptiveStochasticGradientDescent')
	GridSpacing = kwargs.get('GridSpacing')

	HistogramMatch = kwargs.get('HistogramMatch', False)
	IntensityNorm = kwargs.get('IntensityNorm', False)
	GradientMagnitude = kwargs.get('GradientMagnitude', False)
	Blurring = kwargs.get('Blurring', False)
	Sigma = kwargs.get('Sigma', 2)
	EdgeMask = kwargs.get('EdgeMask')
	ReflectionsMasks = kwargs.get('ReflectionsMasks')

	if EdgeMask is not None:
		if ReflectionsMasks is not None:
			GlobalMask = np.logical_or(EdgeMask > 0, ReflectionsMasks > 0).astype(np.uint8)
			GlobalMask = np.invert(GlobalMask) ## SimpleITK uses invert logic
		else:
			GlobalMask = EdgeMask
			GlobalMask = np.invert(GlobalMask) ## SimpleITK uses invert logic
	elif ReflectionsMasks is not None:
		GlobalMask = ReflectionsMasks
		GlobalMask = np.invert(GlobalMask) ## SimpleITK uses invert logic
	else:
		GlobalMask = None

	## Handle cropping
	Cropping = kwargs.get('Cropping', 0)
	if Cropping!=0:
		im_static = im_static[Cropping:(-1*Cropping), Cropping:(-1*Cropping)]
		im_shifted = im_shifted[Cropping:(-1*Cropping), Cropping:(-1*Cropping)]
		if GlobalMask is not None:
			GlobalMask = GlobalMask[Cropping:(-1*Cropping), Cropping:(-1*Cropping)]

	# Print the configuration
	if TwoStage:
		print(f'SimpleElastix: Two-Stage Registration (Affine -> BSpline) with GridSpacing = {GridSpacing}')
	else:
		print(f'SimpleElastix: Transform = {Transform}, Optimizer = {Optimizer}, Metric = {Metric}')

	t0 = time.time()
	
	# New keyword argument for this specific debugging purpose
	return_all_maps = kwargs.get('return_all_maps', False)

	# Convert images to float32
	im_static_orig = im_static.astype(np.float32)
	im_shifted_orig = im_shifted.astype(np.float32)

	# Create copies for preprocessing
	im_static_proc = np.copy(im_static_orig)
	im_shifted_proc = np.copy(im_shifted_orig)

	# Optional preprocessing
	if IntensityNorm:
		# Z-score normalization
		im_static_proc = (im_static_proc - np.mean(im_static_proc)) / np.std(im_static_proc)
		im_shifted_proc = (im_shifted_proc - np.mean(im_shifted_proc)) / np.std(im_shifted_proc)

	if Blurring:
		im_static_proc = gaussian_filter(im_static_proc, sigma=Sigma)
		im_shifted_proc = gaussian_filter(im_shifted_proc, sigma=Sigma)

	# Convert to SimpleITK
	im_static_se = _sitk.GetImageFromArray(im_static_proc)
	im_shifted_se = _sitk.GetImageFromArray(im_shifted_proc)

	if GradientMagnitude:
		# Apply gradient magnitude to the preprocessed images
		im_static_se = _sitk.GradientMagnitude(im_static_se)
		im_shifted_se = _sitk.GradientMagnitude(im_shifted_se)

	if HistogramMatch:
		matcher = _sitk.HistogramMatchingImageFilter()
		matcher.SetNumberOfHistogramLevels(256)
		matcher.SetNumberOfMatchPoints(50)
		matcher.ThresholdAtMeanIntensityOn()
		im_shifted_se = matcher.Execute(im_shifted_se, im_static_se)

	elastixImageFilter = _sitk.ElastixImageFilter()
	if not Verbose:
		elastixImageFilter.LogToConsoleOff()
		

	elastixImageFilter.SetFixedImage(im_static_se)
	elastixImageFilter.SetMovingImage(im_shifted_se)

	if GlobalMask is not None:
		# Ensure correct type and values
		if GlobalMask.dtype != np.uint8:
			GlobalMask = GlobalMask.astype(np.uint8)
		GlobalMask[GlobalMask > 0] = 1 # ensure strictly binary

		# Ensure same shape
		if GlobalMask.shape != im_static_orig.shape:
			raise ValueError(f"GlobalMask shape {GlobalMask.shape} does not match image shape {im_static_orig.shape}")

		# Create SITK mask with same geometry
		mask_se = _sitk.GetImageFromArray(GlobalMask)
		mask_se.CopyInformation(im_static_se) # match origin, spacing, direction

		elastixImageFilter.SetFixedMask(mask_se)
		elastixImageFilter.SetMovingMask(mask_se)

	# Set up the parameter map(s) based on the registration mode
	if TwoStage:
		# Create a list of parameter maps for a two-stage registration
		parameterMap_affine = _sitk.GetDefaultParameterMap('affine')
		parameterMap_bspline = _sitk.GetDefaultParameterMap('bspline')

		# Set the new grid spacing parameter
		if GridSpacing is not None:
			parameterMap_bspline['FinalGridSpacingInPhysicalUnits'] = [str(GridSpacing)]

		# Set the user-defined parameters for the second stage
		parameterMap_bspline['Metric'] = [Metric]
		parameterMap_bspline['Optimizer'] = [Optimizer]
		MaximumNumberOfIterations = kwargs.get('MaximumNumberOfIterations')
		if MaximumNumberOfIterations is not None:
			parameterMap_affine['MaximumNumberOfIterations'] = [str(MaximumNumberOfIterations)]
			parameterMap_bspline['MaximumNumberOfIterations'] = [str(MaximumNumberOfIterations)]

		parameterMap = [parameterMap_affine, parameterMap_bspline]
	elif Affine:
		parameterMap = _sitk.GetDefaultParameterMap('affine')
	else:
		parameterMap = _sitk.GetDefaultParameterMap('translation')
		parameterMap['Metric'] = [Metric]
		parameterMap['Transform'] = [Transform]
		parameterMap['Optimizer'] = [Optimizer]
		# Set the new grid spacing if it's a non-affine transform
		if GridSpacing is not None:
			if 'BSplineTransform' in Transform or 'SplineKernelTransform' in Transform:
				parameterMap['FinalGridSpacingInPhysicalUnits'] = [str(GridSpacing)]


	if not TwoStage and not Affine:
		# These are set by default for TwoStage and Affine
		MaximumNumberOfIterations = kwargs.get('MaximumNumberOfIterations')
		if MaximumNumberOfIterations is not None:
			parameterMap['MaximumNumberOfIterations'] = [str(MaximumNumberOfIterations)]

	elastixImageFilter.SetParameterMap(parameterMap)
	result = elastixImageFilter.Execute()

	# Apply the transform to the original, un-preprocessed image
	transformixImageFilter = _sitk.TransformixImageFilter()
	# This line was added to suppress log output from transformix
	if not Verbose:
		transformixImageFilter.LogToConsoleOff()

	transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
	transformixImageFilter.SetMovingImage(_sitk.GetImageFromArray(im_shifted_orig))
	result_orig = transformixImageFilter.Execute()

	# Save the transform
	transformParameterMap = elastixImageFilter.GetTransformParameterMap()

	im_coregistered = _sitk.GetArrayFromImage(result_orig)

	if Blurring:
		im_coregistered = gaussian_filter(im_coregistered, sigma=Sigma)

	return im_coregistered, transformParameterMap




def SaveFramesSweeps(Hypercube_all, SavingPath, Name, NameSub, **kwargs):
	"""
	Function that takes all hypercubes (where hypercube computed from specific sweepw were kept individually 
	instead of averaged) and saves frames as individual images. 
	All images are saved in folder called {Name}_{NameSub}_RawFrames inside SavingPath, and images from sweep i are
	saved in the folder Sweep{s}.

	Modified to allow expanded hypercubes that include individual (non-averaged) frames within plateaus.

	Input:
		- Hypercube_all : All the hypercubes computed from all sweeps. Shape (Nsweeps, Nwav, Y, X) or (Nsweeps, Nwav, Nframes, Y, X)
		- SavingPath : Where to save all results (generic) (expected to end with '/')
		- Name : General name of the data (i.e. patient)
		- NameSub : Specific name for the hypercubes (i.e. lesion)
		- kwargs:
			- Sweeps [int, int, ...] : If indicated, which sweeps to save
			- Frames [int] : If indicated, which frames inside a given sweep to save
			- Help


	Outputs:
		- All frames saved as png images in individual folders

	"""

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(SaveFramesSweeps))
		return 

	hypercube_shape = Hypercube_all.shape
	if len(hypercube_shape)==4:
		Nsweeps, Nwav, YY, XX = Hypercube_all.shape
	else:
		Nsweeps, Nwav, Nframes, YY, XX = Hypercube_all.shape
		# 19, 16, 3, 1004, 1155

	Sweeps = kwargs.get('Sweeps')
	if Sweeps is None:
		Sweeps = [i for i in range(0, Nsweeps)]
	else:
		print(f'Only saving sweeps {Sweeps}')

	Frames = kwargs.get('Frames')
	if Frames is None:
		Frames = [i for i in range(0, Frames)]
	else:
		print(f'Only saving frames {Frames}')
	
	GeneralDirectory = f'{Name}_{NameSub}_RawFrames'
	try:
		os.mkdir(f'{SavingPath}{GeneralDirectory}')
	except FileExistsError:
		pass
	for s in range(0,Nsweeps):
		if s in Sweeps:
			DirectoryName = f'{SavingPath}{GeneralDirectory}/Sweep{s}'
			try:
				os.mkdir(DirectoryName)
			except FileExistsError:
				pass

			if len(hypercube_shape)==4:
				hypercube = Hypercube_all[s,:,:,:]
				for w in range(0,Nwav):
					frame = hypercube[w,:,:]/(np.amax(hypercube[w,:,:]))*255
					im = Image.fromarray(frame)
					im = im.convert("L")
					im.save(f'{SavingPath}{GeneralDirectory}/Sweep{s}/Im{w}.png')
			else:
				hypercube = Hypercube_all[s,:,:,:,:]
				for w in range(0,Nwav):
					for f in range(0,Nframes):
						if f in Frames:
							frame = hypercube[w,f,:,:]/(np.amax(hypercube[w,f,:,:]))*255
							im = Image.fromarray(frame)
							im = im.convert("L")
							im.save(f'{SavingPath}{GeneralDirectory}/Sweep{s}/Im{w}_frame{f}.png')



def GetNMI(Data, **kwargs):
	"""
	Copmutes the normalised mutual information of a hypercube.

	Inputs:
		- Data : Hypercube, shape N, Y, X
		- kwargs:
			- Help

	Outputs:
		- NMI_average, NMT_vs_reference, NMI_pairwise

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetNMI))
		return 0,0,0


	cube = Data 
	N, H, W = cube.shape
	
	#  Contrast‑normalise each slice so illumination changes between wavelengths do not bias the histogram.
	cube_norm = cube.astype(np.float32).copy()
	for i in range(N):
		sl = cube_norm[i]
		sl -= sl.min()                 # shift to zero
		rng = np.ptp(sl)
		cube_norm[i] = sl / (rng + 1e-7)  # scale to [0, 1]
		
	# Mean NMI against a single reference wavelength
	ref_idx = 1 #N // 2                 # pick the central wavelength as reference
	ref = cube_norm[ref_idx]
	bins = 128                       # histogram bins; 128–256 is typical for 16‑bit data
#     bins = 64                       # histogram bins; 128–256 is typical for 16‑bit data
	nmi_vs_ref = np.empty(N, dtype=np.float32)
	for i in range(N):
		nmi_vs_ref[i] = nmi(cube_norm[i], ref, bins=bins)
	mean_nmi = nmi_vs_ref.mean()
#     print(f"\nMean NMI vs reference slice λ[{ref_idx}] = {mean_nmi:.4f}")
	
	# Pairwise NMI
	pairwise_nmi = np.empty(N - 1, dtype=np.float32)
	for i in range(N - 1):
		pairwise_nmi[i] = nmi(cube_norm[i], cube_norm[i + 1], bins=bins)

	return mean_nmi, nmi_vs_ref, pairwise_nmi


def GetHypercubeForRegistration(Nsweep, Nframe, Path, EdgePos, Wavelengths_list, **kwargs):
	"""
	Function that generates a hypercube for co-registration. 
	Specifically, it allows to select a single specific frame to run the co-registration.

	Inputs:
		- Nsweep (int) : Which sweep in the dataset to use
		- Nframe (int or [int, int, ...]) : Which frame(s), within this sweep, to use (clearest one).
			If more than one frame indicated, the code will compute a hypercube for each integer indicated and then
			concatenate all hypercubes
		- Path : Where the data is located
		- EdgePos : Positions of the start of each sweep
		- Wavelengths_list
		- kwargs:
			- Help
			- Buffer = 9 : How many frames to skip at the start and end of a sweep

	Outputs:
		- Hypercube for registration

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetHypercubeForRegistration))
		return 0

	Buffer = kwargs.get('Buffer')
	if Buffer is None:
		Buffer = 9
		print(f'Setting Buffer to default {Buffer}')
	Hypercube_all, Dark_all = HySE.ComputeHypercube(Path, EdgePos, Wavelengths_list, Buffer=Buffer, Average=False, 
													Order=False, Help=False, SaveFig=False, SaveArray=False, Plot=False, ForCoRegistration=True)
	if isinstance(Nframe, int):
		HypercubeForRegistration = Hypercube_all[Nsweep,:,Nframe,:,:]
	elif isinstance(Nframe, list):
		HypercubeForRegistration = []
		for i in range(0,len(Nframe)):
			HypercubeForRegistration_sub = Hypercube_all[Nsweep,:,Nframe[i],:,:]
			if i==0:
				HypercubeForRegistration = HypercubeForRegistration_sub
			else:
				HypercubeForRegistration = np.concatenate((HypercubeForRegistration, HypercubeForRegistration_sub), axis=0)
	else:
		print(f'Nframe format not accepted.')
		HypercubeForRegistration=0
	return HypercubeForRegistration


def CoRegisterHypercube(RawHypercube, Wavelengths_list, **kwargs):
	"""

	Apply Simple Elastix co-registration to all sweep

	Uses CoRegisterImages

	Input:
		- RawHypercube : To co-registrate. Shape [N, Y, X]
		- Wavelengths_list
		- kwargs:
			- Help
			- Cropping = 0
			- Order = False: Whether to order the coregistered image (based on Wavelenghts_list)
			- Static_Index = 0: Which image is set as the static one (others are registered to it)
			- SaveHypercube
			- PlotDiff = False. If True, plots differences between static, moving and registered images
			- SavingPath. If PlotDiff or SaveHypercbybe is True, where to save the data/figure


	Outputs:
		- Hypercube_Coregistered
		- Coregistration_Transforms

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(CoRegisterHypercube))
		return 0, 0

	PlotDiff = kwargs.get('PlotDiff', False)

	SavingPath = kwargs.get('SavingPath')
	if SavingPath is None:
		SavingPath = ''
		if PlotDiff:
			print(f'PlotDiff has been set to True. Indicate a SavingPath.')

	Static_Index = kwargs.get('Static_Index')
	if Static_Index is None:
		Static_Index = 0
		print(f'Static index set to default {Static_Index}')


	SaveHypercube = kwargs.get('SaveHypercube', True)
	if SaveHypercube:
		print(f'Saving Hypercube')

	Order = kwargs.get('Order', False)

	## Handle cropping
	Cropping = kwargs.get('Cropping', 0)
	if Cropping!=0:
		print(f'Image will be cropped by {Cropping} on all sides.')
	
	t0 = time.time()
	(NN, YY, XX) = RawHypercube.shape


	## Sort Wavelengths
	order_list = np.argsort(Wavelengths_list)
	Wavelengths_sorted = Wavelengths_list[order_list]

	Hypercube = []
	AllTransforms = []

	## Define static image
	im_static = RawHypercube[Static_Index, :,:]

	for c in range(0, NN):
		if c==Static_Index:
			print(f'Static Image')
			im = RawHypercube[c,Cropping:(-1*Cropping),Cropping:(-1*Cropping)]
			# print(f'c={c}: static_im.shape = {im.shape}')
			Hypercube.append(im)
			AllTransforms.append(0)
		else:
			print(f'Working on: {c+1} /{NN}')
			im_shifted = RawHypercube[c, :,:]
			im_coregistered, coregister_transform = CoRegisterImages(im_static, im_shifted, **kwargs) #, **kwargs
			# print(f'c={c}: im_coregistered.shape = {im_coregistered.shape}')
			Hypercube.append(im_coregistered)
			AllTransforms.append(coregister_transform)

			if PlotDiff:
				if '.png' in SavingPath:
					NameTot = SavingPath.split('/')[-1]
					Name = NameTot.replace('.png', '')+f'_{c}.png'
					SavingPathWithName = SavingPath.replace(NameTot, Name)
				else:
					Name = f'_{c}_CoRegistration.png'
					SavingPathWithName = SavingPath+Name
				HySE.UserTools.PlotCoRegistered(im_static, im_shifted, im_coregistered, SavePlot=True, SavingPathWithName=SavingPathWithName)


	tf = time.time()
	# print(f'Hypercube size: len = {len(Hypercube)}')
	# for i in range(0,len(Hypercube)):
		# print(f'i = {i}: {Hypercube[i].shape}')
	Hypercube = np.array(Hypercube)
	## Calculate time taken
	time_total = tf-t0
	minutes = int(time_total/60)
	seconds = time_total - minutes*60
	print(f'\n\n Co-registration took {minutes} min and {seconds:.0f} s in total\n')

	## Sort hypercube according to the order_list
	## Ensures wavelenghts are ordered from blue to red
	if Order:
		Hypercube_sorted = []
		AllTransforms_sorted = []
		for k in range(0,Hypercube.shape[0]):
			Hypercube_sorted.append(Hypercube[order_list[k]])
			AllTransforms_sorted.append(AllTransforms[order_list[k]])
		Hypercube_sorted = np.array(Hypercube_sorted)
	else:
		Hypercube_sorted = np.array(Hypercube)
		AllTransforms_sorted = AllTransforms


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

	return Hypercube_sorted, AllTransforms_sorted




def ApplyTransform(Frames, Transforms, **kwargs):
	'''
	Functions that inputs the list of all transforms generated by HySE.CoRegisterHypercube() and applies 
	it to new frames.
	Note that the transforms can be saved as .txt files as well

	Inputs:
		- Frames : Array, shape (N, Y, X)
		- Transforms : List, length N (one transform per 2D frame)
		- kwargs :
			- Help
			- Verbose = False : If true, prints default console output from SimpleITK

	Outputs:
		- TransformedFrames
	'''
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(ApplyTransform))
		return 0
	Verbose = kwargs.get('Verbose', False)

	(Nwav, Y, X) = Frames.shape
	if Nwav!=len(Transforms):
		print(f'The number of frames to register does not match the number of transforms provided')
		return 0
	
	TransformedFrames = []
	for i in range(0, Nwav):
		print(f'Transforming image {i+1}/Nwav')
		im_shifted = Frames[i,:,:]
		transform = Transforms[i]
		
		if transform==0:
			# this is the static frame (transform==0) – nothing to apply
			print(f'Static image')
			TransformedFrames.append(im_shifted)
			continue
		else:
			im_shifted_se = _sitk.GetImageFromArray(im_shifted)
			transformixImageFilter = _sitk.TransformixImageFilter()
			if Verbose==False:
				transformixImageFilter.LogToConsoleOff()
			transformixImageFilter.SetMovingImage(im_shifted_se)
			transformixImageFilter.SetTransformParameterMap(transform)
			result = transformixImageFilter.Execute()
			im_registered = _sitk.GetArrayFromImage(result)
			TransformedFrames.append(im_registered)
		
	TransformedFrames = np.array(TransformedFrames)
	return TransformedFrames


def SaveTransforms(AllTransforms, TransformsSavingPath, **kwargs):
	'''
	Function that saves all the transforms output from the coregistration pipeline (AllTransforms) as txt files.
	
	Inputs:
		- AllTransforms (Output from CoRegisterHypercube())
		- TransformsSavingPath : Where to save files (do not add extension)
		- kwargs
			- Help
	Ouput:
	
	'''
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(SaveTransforms))
		return 
	
	N = len(AllTransforms)
	for n in range(0,N):
		transform = AllTransforms[n]
		if transform==0:
#             print('Static image')
			pass
		else:
			for i in range(0,len(transform)):
				_sitk.WriteParameterFile(transform[i], f'{TransformsSavingPath}_Frame{n}_{i}.txt')
			
	
def LoadTransforms(TransformsPath, **kwargs):
	'''
	Function that loads previously saved coregistration transforms to apply to new frames.
	Looks for all the txt files in the indicated path, and checks for the number of transforms applied
	to each frame.
	Assumes that the files were generated by SaveTransforms() and that TransformPath is the same path
	indicated in SaveTransforms() when creating the files.
	
	Input:
		- TransformsPath: Path where the files are located. Assumes .txt files and format
			of the shape Frame{n}_{i}, where n is the frame and i is the transform
			For TwoStage tranform, i=0 is the affine transform and i=1 is the BSpline transform
		- kwargs:
			- Help
			
	Output:
		- AllTransforms : list of (lists) containting all the transforms. 
			len(AllTransforms) = number of frames. 
			If more than one transform is applied per frame (TwoStage), each element in AllTransforms will
			be a list containing all the transforms applied to this given frame.
	
	'''
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(LoadTransforms))
		return 
	Files = natsorted(glob.glob(TransformsPath+'*.txt'))

	TransformInfo = []
	for i in range(0,len(Files)):
		file = Files[i]
		Name = file.split('/')[-1]
		Info = Name.split('Frame')[-1].replace('.txt','')
		(Frame, t) = Info.split('_')
		TransformInfo.append(int(t))

	NTransformsPerFrame = np.amax(TransformInfo)
	
	AllTransforms = []
	for i in range(0,len(Files)):
		file = Files[i]
		Name = file.split('/')[-1]
		Info = Name.split('Frame')[-1].replace('.txt','')
		transform = _sitk.ReadParameterFile(file)
		(Frame, t) = Info.split('_')
		if int(t)==0:
			TransformsFrameSub = []
		TransformsFrameSub.append(transform)
		if int(t)==NTransformsPerFrame:
			AllTransforms.append(TransformsFrameSub)
		
	return AllTransforms
			

