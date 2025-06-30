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





def GetCoregisteredHypercube(vidPath, EdgePos, Nsweep, Wavelengths_list, **kwargs):
	"""

	~~ Co-registration only ~~

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
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetCoregisteredHypercube))
		return 0

	## ImportDatafor the sweep
	DataSweep = HySE.Import.GetSweepData_FromPath(vidPath, EdgePos, Nsweep, **kwargs)
	## Compute Hypercube
	Hypercube = SweepCoRegister(DataSweep, Wavelengths_list, **kwargs)
	return Hypercube




def SweepCoRegister_WithNormalisation(DataSweep, WhiteHypercube, Dark, Wavelengths_list, **kwargs):
	"""

	~~ Co-registration with normalisation and masking ~~

	NB: Old function, normalisation and masking not optimal. Never tested with wavelength mixing.
		New function to be written.

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
	Help = kwargs.get('Help', False)
	if Help:
		# print(info)
		print(inspect.getdoc(SweepCoRegister_WithNormalisation))
		return 0

	AllIndices = [DataSweep[i].shape[0] for i in range(0,len(DataSweep))]
	MaxIndex = np.amax(AllIndices)
	MinIndex = np.amin(AllIndices)
	# print(AllIndices)
	# print(MaxIndex)

	Buffer = kwargs.get('Buffer', 6)

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
		print(f'PlotDiff has been set to True. Indicate a SavingPath.')
	

	Plot_PlateauList = kwargs.get('Plot_PlateauList')
	if Plot_PlateauList is None:
		Plot_PlateauList = [5]
		print(f'Set Plot_PlateauList and Plot_Index to set images to plot')
	else:
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
			print(f'	Seeting it to {PlotIndex}')



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

	## Define static image
	im_static = DataSweep[ImStatic_Plateau][ImStatic_Index,:,:]
	ImStatic_Plateau_sorted = order_list[ImStatic_Plateau]
	im_staticN = HySE.ManipulateHypercube.NormaliseFrames(im_static, WhiteHypercube[ImStatic_Plateau_sorted,:,:], Dark)

	## Loop through all colours (wavelengths)
	print(f'\n Plot_PlateauList = {Plot_PlateauList}, Plot_Index = {Plot_Index}\n')

	for c in range(0, Ncolours):
		if c==8: ## ignore dark
			# print(f'DARK')
			pass
		else:

			print(f'Working on: {c} /{Ncolours}')
			# ImagesTemp = []
			# (NN, YY, XX) = DataSweep[c].shape

			## Which of the first 3 images if brightest?
			vals = [np.average(DataSweep[c][Buffer+q,:,:]) for q in range(Buffer,Buffer+3)]
			
			offset = np.where(vals==np.amax(vals))[0][0]


			ImagesTemp = []
			(NN, YY, XX) = DataSweep[c].shape
			# ## Averaging All frames
			# for i in range(Buffer,NN-Buffer):
			## Averaing brightest frames
			for i in range(Buffer+offset,NN-Buffer,3):
				im_shifted = DataSweep[c][i,:,:]
				if c>=8: ## the hypercube does not include dark in the middle
					hypercube_index = order_list[c-1] ## hypercube is already sorted
					im_white = WhiteHypercube[hypercube_index,:,:]
				else:
					hypercube_index = order_list[c]
					im_white = WhiteHypercube[hypercube_index,:,:]

				im_shiftedN = HySE.ManipulateHypercube.NormaliseFrames(im_shifted, im_white, Dark)

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
								HySE.UserTools.PlotCoRegistered(im_static, im_shifted, im_coregistered, SavePlot=True, SavingPathWithName=SavingPathWithName)

			
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

	~~ Used in SweepCoRegister_WithNormalisation ~~

	NB: Old function, normalisation and masking not optimal. Never tested with wavelength mixing.
		New function to be written.

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
			print(f'	Seeting it to {PlotIndex}')


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

			# ## Average all frames
			# for i in range(Buffer,NN-Buffer):
			## Average only brightes frames:
			for i in range(Buffer+offset,NN-Buffer,3):
				im_shifted = DataSweep[c][i,:,:]
				im_coregistered, shift_val, time_taken = CoRegisterImages(im_static, im_shifted)
				ImagesTemp.append(im_coregistered)

				if c<8:
					print(f'      c={c}, i={i}: {Wavelengths_list[c]} nm, avg {np.average(im_shifted)}, shift val {shift_val:.2f}, {time_taken:.2f} s')
				else:
					print(f'      c={c}, i={i}: {Wavelengths_list[c-1]} nm, avg {np.average(im_shifted)}, shift val {shift_val:.2f}, {time_taken:.2f} s')

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

			
			ImagesTemp = np.array(ImagesTemp)

			fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(15, 10))
			# DataSweep[c][i,:,:]
			m, M = np.amin(DataSweep[c][:,:,:]), np.amax(DataSweep[c][:,:,:])

			ax[0,0].imshow(DataSweep[c][0,:,:], cmap='gray', vmin=m, vmax=M)
			ax[0,0].set_title('frame 0 - orig')
			# ax[0,0].imshow(im_static, cmap='gray', vmin=m, vmax=M)
			# ax[0,0].set_title('im_static')
			kk = int(len(ImagesTemp)/2)
			ax[0,1].imshow(DataSweep[c][kk,:,:], cmap='gray', vmin=m, vmax=M)
			ax[0,1].set_title(f'frame {kk} - orig')
			ax[0,2].imshow(DataSweep[c][-1,:,:], cmap='gray', vmin=m, vmax=M)
			ax[0,2].set_title('frame -1 - orig')

			ax[0,3].imshow(im_static, cmap='gray', vmin=m, vmax=M)
			ax[0,3].set_title('im_static')

			# m, M = np.amin(ImagesTemp), np.amax(ImagesTemp)
			ax[1,0].imshow(ImagesTemp[0, :, :], cmap='gray', vmin=m, vmax=M)
			ax[1,0].set_title('frame 0 - CR')
			kk = int(len(ImagesTemp)/2)
			ax[1,1].imshow(ImagesTemp[kk, :, :], cmap='gray', vmin=m, vmax=M)
			ax[1,1].set_title(f'frame {kk} - CR')
			ax[1,2].imshow(ImagesTemp[-1, :, :], cmap='gray', vmin=m, vmax=M)
			ax[1,2].set_title('frame -1 - CR')

			
			for w in range(0,4):
				for p in range(0,2):
					ax[p,w].set_xticks([])
					ax[p,w].set_yticks([])

			fig.delaxes(ax[1,3])
			plt.tight_layout()
			plt.savefig(f'/Users/iracicot/Library/CloudStorage/OneDrive-UniversityofCambridge/Data/HySE/Patient3_20250114/Flat/test/c{c}.png')
			plt.close()

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





def CoRegisterImages(im_static, im_shifted, **kwargs):
	"""
	Basic function that co-registers a shifted image to a defined static image.
	It uses SimpleElastix funtions. 
	Lines of code can be commented 


	Inputs:
		- im_static
		- im_shifted
		- kwargs:
			- Affine
			- Verbose
			- MaximumNumberOfIterations: integer (e.g. 500)
			- Metric (default 'AdvancedMattesMutualInformation', also 'NormalizedCorrelation' and 'AdvancedKappaStatistic')
			- Optimizer (default 'AdvancedMattesMutualInformation')
			- Transform (default 'BSplineTransform'). Also:
				Global (Parametric) Transforms:
				- 'TranslationTransform': Only accounts for shifts (translations)
				- 'Euler2DTransform': Rigid transformations, including translation and rotation
				- 'VersorTransform': Similar to Euler, rotations are represented by a versor (quaternion), which can be more numerically stable for large rotations.
				- 'Similarity2DTransform': Adds isotropic scaling to the rigid transformations (translation, rotation, and uniform scaling).
				- 'ScaleTransform': Allows for anisotropic scaling (different scaling factors along each axis)
				- 'AffineTransform': A general linear transformation that includes translation, rotation, scaling (isotropic and anisotropic), and shearing
				Deformable (Non-Parametric) Transforms
				- 'BSplineTransform': uses a sparse regular grid of control points to define a smooth, non-rigid deformation field
				- 'DisplacementFieldTransform': epresents the transformation as a dense grid of displacement vectors, where each pixel has a corresponding vector indicating its movement. 
				This offers the highest flexibility but can be computationally more expensive and may require more memory.

	Outputs:
		- im_coregistered
		- shift_val: estiimate of the maximal shift
		- time_taken


	"""
	if _sitk is None:
		raise ImportError(
			"This function requires SimpleITK. "
			"Install it with `pip install SimpleITK` or "
			"`pip install HySE[registration]`."
		)

		
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(CoRegisterImages))
		return 0,0,0

	## If we don't expect complex deformations, set transform to affine
	## To limit unwanted distortions
	Affine = kwargs.get('Affine', False)
	Verbose = kwargs.get('Verbose', False)
	Metric = kwargs.get('Metric', 'AdvancedMattesMutualInformation')
	Transform = kwargs.get('Transform', 'BSplineTransform')
	Optimizer = kwargs.get('Transform', 'AdaptiveStochasticGradientDescent')
		
		
	t0 = time.time()
	## Convert the numpy array to simple elestix format
	im_static_se = _sitk.GetImageFromArray(im_static)
	im_shifted_se = _sitk.GetImageFromArray(im_shifted)

	## Create object
	elastixImageFilter = _sitk.ElastixImageFilter()

	## Turn off console
	if Verbose==False:
		elastixImageFilter.LogToConsoleOff()

	## Set image parameters
	elastixImageFilter.SetFixedImage(im_static_se)
	elastixImageFilter.SetMovingImage(im_shifted_se)

	## Set transform parameters
	if Affine:
		parameterMap = _sitk.GetDefaultParameterMap('affine')
	else:
		parameterMap = _sitk.GetDefaultParameterMap('translation')
		## Select metric robust to intensity differences (non uniform)
		parameterMap['Metric'] = [Metric] 
		## Select Bspline transform, which allows for non rigid and non uniform deformations
		parameterMap['Transform'] = [Transform]

	## Tried those parameters, on Macbeth chart data (not moving), did not have significant impact
	# parameterMap['AutomaticTransformInitialization'] = ['true']
	# parameterMap['AutomaticTransformInitializationMethod'] = ['CenterOfGravity']
	# parameterMap['HistogramMatch'] = ['true']
	# parameterMap['BSplineRegularizationOrder'] = ['11'] ## default 

	## Parameters to play with if co-registration is not optimal:

#         # Controls how long the optimizer runs
	parameterMap['Optimizer'] = [Optimizer]
	MaximumNumberOfIterations = kwargs.get('MaximumNumberOfIterations')
	if MaximumNumberOfIterations is not None:
		parameterMap['MaximumNumberOfIterations'] = [str(MaximumNumberOfIterations)] 

	#     # You can try different metrics like AdvancedMattesMutualInformation, NormalizedCorrelation, 
	#     # or AdvancedKappaStatistic for different registration scenarios.
	# parameterMap['Metric'] = ['AdvancedMattesMutualInformation']
	#     # Adjust the number of bins used in mutual information metrics
	# parameterMap['NumberOfHistogramBins'] = ['32']
	#     # Change the optimizer to AdaptiveStochasticGradientDescent for potentially better convergence
	# parameterMap['Optimizer'] = ['AdaptiveStochasticGradientDescent']
	#     # Controls the grid spacing for the BSpline transform
	# parameterMap['FinalGridSpacingInPhysicalUnits'] = ['10.0']
	#     # Refines the BSpline grid at different resolutions.
	# parameterMap['GridSpacingSchedule'] = ['10.0', '5.0', '2.0']
	#     # Automatically estimate the scales for the transform parameters.
	# parameterMap['AutomaticScalesEstimation'] = ['true']
	#     # Controls the number of resolutions used in the multi-resolution pyramid. 
	#     # A higher number can lead to better registration at the cost of increased computation time.
	# parameterMap['NumberOfResolutions'] = ['4']
	#     # Automatically initializes the transform based on the center of mass of the images.
	# parameterMap['AutomaticTransformInitialization'] = ['true']
	#     # Controls the interpolation order for the final transformation.
	# parameterMap['FinalBSplineInterpolationOrder'] = ['3']

#         # Adjust the maximum step length for the optimizer
#     parameterMap['MaximumStepLength'] = ['4.0']
#         # Use more samples for computing gradients
#     parameterMap['NumberOfSamplesForExactGradient'] = ['10000']
#         # Specify the grid spacing in voxels for the final resolution.
#     parameterMap['FinalGridSpacingInVoxels'] = ['8.0']
#         # Defines the spacing of the sampling grid used during optimization.
#     parameterMap['SampleGridSpacing'] = ['2.0']


	## If required, set maximum number of iterations
#     parameterMap['MaximumNumberOfIterations'] = ['500']
	elastixImageFilter.SetParameterMap(parameterMap)

	## Execute
	result = elastixImageFilter.Execute()
	## Convert result to numpy array
	im_coregistered = _sitk.GetArrayFromImage(result)
	t1 = time.time()

	## Find time taken:
	time_taken = t1-t0

	## Get an idea of difference
	shift_val = np.average(np.abs(np.subtract(im_static,im_coregistered)))

	## return 
	return im_coregistered, shift_val, time_taken



