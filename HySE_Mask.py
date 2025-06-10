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
import HySE_ManipulateHypercube


def GetStandardMask(WhiteCalibration, **kwargs):
	threshold = kwargs.get('threshold', 1)
	Calibration_avg = np.average(np.average(WhiteCalibration, axis=1), axis=1)
	max_idx = np.where(Calibration_avg==np.amax(Calibration_avg))[0][0]
	Mask = WhiteCalibration[max_idx, :,:] < threshold
	return Mask

def ConvertMaskToBinary(mask):
	mask_binary = 1-mask*1
	return mask_binary.astype('uint8')

def BooleanMaskOperation(bool_white, bool_wav):
	bool_result = False
	if bool_white!=bool_wav:
		if bool_wav==1:
			bool_result = True
	return bool_result

def TakeWavMaskDiff(mask_white, mask_shifted):
	vectorized_function = np.vectorize(BooleanMaskOperation)
	result = vectorized_function(mask_white, mask_shifted)
	return result

def CombineMasks(mask_white, mask_shifted):
	mask = np.ma.mask_or(mask_white, mask_shifted)
	mask = mask*1
	mask = mask.astype('uint8')
	return mask
	

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
	Help = kwargs.get('Help', False)
	if Help:
		print(info)
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
	Help = kwargs.get('Help', False)
	if Help:
		print(info)
		return 0

	Affine = kwargs.get('Affine', False)
	Verbose = kwargs.get('Verbose', False)

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
#             print(f'Static: Boolean mask. Converting to binary')
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
#             print(f'Moving: Boolean mask. Converting to binary')
			MovingMask = ConvertMaskToBinary(MovingMask)
		elif type(MovingMask[0,0])!=np.uint8:
			print(f'MovingMask is neither in uint8 or boolean format, code won\'t run')
		MovingMask_se = sitk.GetImageFromArray(MovingMask)
		elastixImageFilter.SetMovingMask(MovingMask_se)
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




def SweepCoRegister_MaskedWithNormalisation(DataSweep, WhiteHypercube, Dark, Wavelengths_list, **kwargs):
	info="""
	Apply Simple Elastix co-registration to all sweep. Keeping the same static image, as specified by user (or set automatically).

	Input:
		- DataSweep: List of 3D arrays. Each element in the list contains all frames in a plateau (wavelength)
		- WhiteHypercube: White reference hypercube (3D array, assumed sorted)
		- Dark: 2D array
		- Wavelengths_list: not sorted
		- kwargs 
			- Buffer: sets the numner of frames to ignore on either side of a colour transition
				Total number of frames removed = 2*Buffer (default 6)
			- ImStatic_Wavelength: sets the wavelength (in nm) from which the static image is selected (default closest to 550)
			- ImStatic_Index: sets which frame in the selected plateau (wavelength) as the static image (default 8)
			- LowCutoff: For masking (default False)
			- HighCutoff: For masking (default False)
			- Mask_CombinedAvgCutoff (default 0.01): when the average value of the combined masks is above this cutoff, only high
				cutoff is used for the moving mask in the coregistration
			- SavingPath: Where to save figure (default '')
			- SaveHypercube: whether or not to save the hypercybe and the sorted wavelengths as npz format
				(default True)
			- PlotDiff: Whether to plot figure showing the co-registration (default False)
				If set to True, also expects:
				- Plot_PlateauList: for which plateau(x) to plot figure. Aceepts a list of integers or "All" for all plateau (default 5)
				- Plot_Index: which frame (index) to plot for each selected plateau (default 14)
			- Help: print this help message is True


	Outputs:
		- Normalised and co-registered Hypercube

	"""
	
	Help = kwargs.get('Help', False)
	if Help:
		print(info)
		return 0, 0

	AllIndices = [DataSweep[i].shape[0] for i in range(0,len(DataSweep))]
	MaxIndex = np.amax(AllIndices)
	MinIndex = np.amin(AllIndices)

	## Sort Wavelengths
	order_list = np.argsort(Wavelengths_list)
	Wavelengths_sorted = Wavelengths_list[order_list]

	mask_combined_avg_cutoff = kwargs.get('mask_combined_avg_cutoff', 0.01)
	Buffer = kwargs.get('Buffer', 6)
	LowCutoff = kwargs.get('LowCutoff', False)
	HighCutoff = kwargs.get('HighCutoff', False)

	ImStatic_Index = kwargs.get('ImStatic_Index')
	if not ImStatic_Index:
		ImStatic_Index = 8
		if MinIndex>ImStatic_Index:
			ImStatic_Index = int(MinIndex/2)
			print(f'ImStatic_Index is outside default range. Set to {ImStatic_Index}, please set manually with ImStatic_Index')
	else:
		if ImStatic_Index<5 or ImStatic_Index<Buffer:
			print(f'Careful! You have set ImStatic_Index < 5 or < Buffer ')
			print(f'	This index risks being in the range of unreliable frames too close to a colour transition.')
		if ImStatic_Index>(MinIndex-Buffer):
			print(f'Careful! You have set ImStatic_Index  > (MinIndex - Buffer')
			print(f'	This index risks being in the range of unreliable frames too close to a colour transition.')


	ImStatic_Wavelength = kwargs.get('ImStatic_Wavelength')
	if not ImStatic_Wavelength:
		ImStatic_Wavelength = 550
		StaticWav_index = HySE_UserTools.find_closest(Wavelengths_list, ImStatic_Wavelength)
		StaticWav_index_sorted = HySE_UserTools.find_closest(Wavelengths_sorted, ImStatic_Wavelength)
		ImStatic_Wavelength = Wavelengths_list[StaticWav_index]
		print(f'ImStatic_Wavelength set by default closest to 550, to {ImStatic_Wavelength} nm')
	else:
		StaticWav_index = HySE_UserTools.find_closest(Wavelengths_list, ImStatic_Wavelength)
		StaticWav_index_sorted = HySE_UserTools.find_closest(Wavelengths_sorted, ImStatic_Wavelength)
		ImStatic_Wavelength = Wavelengths_list[StaticWav_index]
		print(f'ImStatic_Wavelength set to {ImStatic_Wavelength} nm')

	print(f'Static image: {ImStatic_Wavelength} nm (index/plateau {StaticWav_index}), index {ImStatic_Index}. Use ImStatic_Wavelength and ImStatic_Index to change it.')
	print(f'   NB: ImStatic_Index refers to the frame number in a given plateau/wavelenght used as initial static image. Not to be confused with array index,')


	PlotDiff = kwargs.get('PlotDiff', False)
	if PlotDiff:
		print(f'PlotDiff set to True. Use \'Plot_PlateauList=[]\' or \'All\' and Plot_Index=int to set')


	SavingPath = kwargs.get('SavingPath')
	if not SavingPath:
		SavingPath = ''
		print(f'PlotDiff has been set to True. Indicate a SavingPath.')


	Plot_PlateauList = kwargs.get('Plot_PlateauList')
	if not Plot_PlateauList:
		Plot_PlateauList = [5]
		print(f'Set Plot_PlateauList and Plot_Index to set images to plot')
	else:
		if isinstance(Plot_PlateauList, int):
			Plot_PlateauList = [Plot_PlateauList]


	Plot_Index = kwargs.get('Plot_Index')
	if not Plot_Index:
		Plot_Index = 14
		print(f'MinIndex = {MinIndex}, MinIndex-Buffer = {MinIndex-Buffer}')
		if Plot_Index>(MinIndex-Buffer):
			Plot_Index = int(MinIndex/2)
			print(f'Plot_Index outside default range. Set to {Plot_Index}, please set manually with Plot_Index')
	else:
		if Plot_Index<Buffer or Plot_Index>(MinIndex-Buffer):
			print(f'PlotIndex is outside the range of indices that will be analysed ({Buffer}, {MinIndex-Buffer})')
			Plot_Index = int(MinIndex/2)
			print(f'	Seeting it to {PlotIndex}')


	

	SaveHypercube = kwargs.get('SaveHypercube', True)
	if SaveHypercube:
		print(f'SaveHypercube set to {SaveHypercube}')


	print(f'Buffer set to {Buffer}')

	t0 = time.time()
	Ncolours = len(DataSweep)-1
	(_, YY, XX) = DataSweep[1].shape

	## Deal with special cases when plateau list is input as string
	if isinstance(Plot_PlateauList, str):
		if Plot_PlateauList=='All':
			Plot_PlateauList = [i for i in range(0,Ncolours)]
		elif Plot_PlateauList=='None':
			Plot_PlateauList = []


	## Define static image
	if StaticWav_index>=8:
		## to account for the fact that the dark is not included in the wavelengths list
		im_static_0 = DataSweep[StaticWav_index-1][ImStatic_Index,:,:].astype('float64')
	else:
		im_static_0 = DataSweep[StaticWav_index][ImStatic_Index,:,:].astype('float64')

	White_static_0 = WhiteHypercube[StaticWav_index_sorted,:,:]
	im_staticN_init = HySE_ManipulateHypercube.NormaliseFrames(im_static_0, White_static_0, Dark) ## HySE_ManipulateHypercube.
	im_staticN_0 = im_staticN_init

	## Loop through all colours (wavelengths)
	print(f'\n Plot_PlateauList = {Plot_PlateauList}, Plot_Index = {Plot_Index}\n')

	Hypercube = np.zeros(WhiteHypercube.shape)
	Hypercube_Masks = np.zeros(WhiteHypercube.shape)

	## Starting from static image to higher wavelengths
	for u in range(0, Ncolours):
#     for u in range(StaticWav_index_sorted, StaticWav_index_sorted+1):
		wav = Wavelengths_sorted[u]
		## Set static image for this wavelength
		im_staticN = im_staticN_0
		## Find wavelnegth index in raw data frame
		c = np.where(Wavelengths_list==wav)[0][0]
		if c>=8:
			c=c+1
		## Find white reference for wavelength
		im_white = WhiteHypercube[u,:,:]
		
		## Now co-register all frames
		ImagesTemp = []
		MasksTemp = []
		(NN, YY, XX) = DataSweep[c].shape
#         print(f'Index range: {Buffer}, {NN-Buffer}')
		for i in range(Buffer,NN-Buffer):
			im_shifted = DataSweep[c][i,:,:].astype('float64')
			## Masks
			mask_white = GetMask(im_white, LowCutoff=LowCutoff, HighCutoff=HighCutoff)
			mask_shifted = GetMask(im_shifted, LowCutoff=0, HighCutoff=HighCutoff)
			mask_combined = TakeWavMaskDiff(mask_white, mask_shifted)
			MasksTemp.append(mask_combined*1)
			mask_combined_avg = np.average(mask_combined)
			
			## Normalise before co-registration
			im_shiftedN = HySE_ManipulateHypercube.NormaliseFrames(im_shifted, im_white, Dark) #HySE_ManipulateHypercube.
			
			## Co-registrate with masks
			## Because I am doing the co-registration frame by frame and therefore am not averaging frames for a given wavelengths,
			## dark values oscillate between ~6 and exactly 0. These 0 values are caught by the mask, which can quickly add to
			## too many masked values for the elastix moving mask option. 
			## When that is the case, only consider the static mask (from the white image)
			if mask_combined_avg>mask_combined_avg_cutoff:
				mask_shifted = GetMask(im_shifted, HighCutoff=HighCutoff)
#                 print(f'Shifted mask: avg: {np.average(mask_shifted):.2f}, HighCutoff = {HighCutoff}')
				im_coregistered = CoRegisterImages_WithMask(im_staticN, im_shiftedN, StaticMask=mask_white, MovingMask=mask_shifted, **kwargs)
#                 im_coregistered = CoRegisterImages_WithMask(im_staticN, im_shiftedN, StaticMask=mask_white, **kwargs)#, MovingMask=mask_shifted, Verbose=True)
			else:
				im_coregistered = CoRegisterImages_WithMask(im_staticN, im_shiftedN, StaticMask=mask_white, MovingMask=mask_combined, **kwargs)#, MovingMask=mask_shifted, Verbose=True)
			
			ImagesTemp.append(im_coregistered)

			## Plot co-registration if requested
			if PlotDiff:
				if c in Plot_PlateauList:
					if '.png' in SavingPath:
						NameTot = SavingPath.split('/')[-1]
						Name = NameTot.replace('.png', '')+f'_Plateau{c}_Index{i}.png'
						SavingPathWithName = SavingPath.replace(NameTot, Name)
					else:
						Name = f'_{wav}nm_Index{i}_CoRegistration.png'
						SavingPathWithName = SavingPath+Name

					if i==Plot_Index:
						HySE_UserTools.PlotCoRegistered(im_staticN, im_shiftedN, im_coregistered, SavePlot=True, SavingPathWithName=SavingPathWithName)

		ImagesTemp = np.array(ImagesTemp)
		MasksTemp = np.array(MasksTemp)
		ImAvg = np.average(ImagesTemp, axis=0)
		MasksAvg = np.round(np.average(MasksTemp, axis=0),0)
		MasksAvg = MasksAvg.astype('uint8')

		Hypercube[u,:,:] = ImAvg
		Hypercube_Masks[u,:,:] = MasksAvg


	## Calculate time taken
	tf = time.time()
	time_total = tf-t0
	minutes = int(time_total/60)
	seconds = time_total - minutes*60
	print(f'\n\n Co-registration took {minutes} min and {seconds:.0f} s in total\n')

	if SaveHypercube:
		if '.png' in SavingPath:
			NameTot = SavingPath.split('/')[-1]
			Name = NameTot.replace('.png', '')+f'_CoregisteredHypercube.npz'
			Name_wav = NameTot.replace('.png', '')+f'_CoregisteredHypercube_wavelengths.npz'
			Name_mask = NameTot.replace('.png', '')+f'_CoregisteredHypercube_masks.npz'
			SavingPathHypercube = SavingPath.replace(NameTot, Name)
			SavingPathWavelengths = SavingPath.replace(NameTot, Name_wav)
			SavingPathMasks = SavingPath.replace(NameTot, Name_mask)
		else:
			Name = f'_CoregisteredHypercube.npz'
			Name_wav = f'_CoregisteredHypercube_wavelengths.npz'
			Name_mask = f'_CoregisteredHypercube_masks.npz'
			SavingPathHypercube = SavingPath+Name
			SavingPathWavelengths = SavingPath+Name_wav
			SavingPathMasks = SavingPath+Name_mask

		np.savez(f'{SavingPathHypercube}', Hypercube)
		np.savez(f'{SavingPathWavelengths}', Wavelengths_sorted)
		np.savez(f'{SavingPathMasks}', Hypercube_Masks)

	return Hypercube, Hypercube_Masks


def SweepRollingCoRegister_MaskedWithNormalisation(DataSweep, WhiteHypercube, Dark, Wavelengths_list, **kwargs):
	info="""
	Apply Simple Elastix co-registration to all sweep. 
	Starts with input static image and then propagates co-registration from this static image, resetting
	a new static image for every frame to ensure that both static and moving images do not have large 
	differences caused by different illumination wavelengths. 

	NB: By using the result from previous co-registrations as static images, this method propagates distortions
	across the hypercube and can lead to nonsensical results.

	Input:
		- DataSweep: List of 3D arrays. Each element in the list contains all frames in a plateau (wavelength)
		- WhiteHypercube: White reference hypercube (3D array, assumed sorted)
		- Dark: 2D array
		- Wavelengths_list: not sorted
		- kwargs 
			- Buffer: sets the numner of frames to ignore on either side of a colour transition
				Total number of frames removed = 2*Buffer (default 6)
			- ImStatic_Wavelength: sets the wavelength (in nm) from which the static image is selected (default closest to 550)
			- ImStatic_Index: sets which frame in the selected plateau (wavelength) as the static image (default 8)
			- LowCutoff: For masking (default False)
			- HighCutoff: For masking (default False)
			- Mask_CombinedAvgCutoff (default 0.01): when the average value of the combined masks is above this cutoff, only high
				cutoff is used for the moving mask in the coregistration
			- SavingPath: Where to save figure (default '')
			- SaveHypercube: whether or not to save the hypercybe and the sorted wavelengths as npz format
				(default True)
			- PlotDiff: Whether to plot figure showing the co-registration (default False)
				If set to True, also expects:
				- Plot_PlateauList: for which plateau(x) to plot figure. Aceepts a list of integers or "All" for all plateau (default 5)
				- Plot_Index: which frame (index) to plot for each selected plateau (default 14)
			- Help: print this help message is True


	Outputs:
		- Normalised and co-registered Hypercube

	"""
	
	Help = kwargs.get('Help', False)
	if Help:
		print(info)
		return 0, 0

	AllIndices = [DataSweep[i].shape[0] for i in range(0,len(DataSweep))]
	MaxIndex = np.amax(AllIndices)
	MinIndex = np.amin(AllIndices)
	# print(AllIndices)
	# print(MaxIndex)

	## Sort Wavelengths
	order_list = np.argsort(Wavelengths_list)
	Wavelengths_sorted = Wavelengths_list[order_list]

	mask_combined_avg_cutoff = kwargs.get('mask_combined_avg_cutoff', 0.01)
	Buffer = kwargs.get('Buffer', 6)

	LowCutoff = kwargs.get('LowCutoff', False)
	HighCutoff = kwargs.get('HighCutoff', False)


	ImStatic_Index = kwargs.get('ImStatic_Index')
	if not ImStatic_Index:
		ImStatic_Index = 8
		if MinIndex>ImStatic_Index:
			ImStatic_Index = int(MinIndex/2)
			print(f'ImStatic_Index is outside default range. Set to {ImStatic_Index}, please set manually with ImStatic_Index')
	else:
		if ImStatic_Index<5 or ImStatic_Index<Buffer:
			print(f'Careful! You have set ImStatic_Index < 5 or < Buffer ')
			print(f'	This index risks being in the range of unreliable frames too close to a colour transition.')
		if ImStatic_Index>(MinIndex-Buffer):
			print(f'Careful! You have set ImStatic_Index  > (MinIndex - Buffer')
			print(f'	This index risks being in the range of unreliable frames too close to a colour transition.')


	ImStatic_Wavelength = kwargs.get('ImStatic_Wavelength')
	if not ImStatic_Wavelength:
		ImStatic_Wavelength = 550
		StaticWav_index = HySE_UserTools.find_closest(Wavelengths_list, ImStatic_Wavelength)
		StaticWav_index_sorted = HySE_UserTools.find_closest(Wavelengths_sorted, ImStatic_Wavelength)
		ImStatic_Wavelength = Wavelengths_list[StaticWav_index]
		print(f'ImStatic_Wavelength set by default closest to 550, to {ImStatic_Wavelength} nm')
	else:
		ImStatic_Wavelength = kwargs['ImStatic_Wavelength']
		StaticWav_index = HySE_UserTools.find_closest(Wavelengths_list, ImStatic_Wavelength)
		StaticWav_index_sorted = HySE_UserTools.find_closest(Wavelengths_sorted, ImStatic_Wavelength)
		ImStatic_Wavelength = Wavelengths_list[StaticWav_index]
		print(f'ImStatic_Wavelength set to {ImStatic_Wavelength} nm')


	print(f'Static image: {ImStatic_Wavelength} nm (index/plateau {StaticWav_index}), index {ImStatic_Index}. Use ImStatic_Wavelength and ImStatic_Index to change it.')
	print(f'   NB: ImStatic_Index refers to the frame number in a given plateau/wavelenght used as initial static image. Not to be confused with array index,')


	PlotDiff = kwargs.get('PlotDiff', False)
	if PlotDiff:
		print(f'PlotDiff set to True. Use \'Plot_PlateauList=[]\' or \'All\' and Plot_Index=int to set')

	SavingPath = kwargs.get('SavingPath')
	if not SavingPath:
		SavingPath = ''
		print(f'PlotDiff has been set to True. Indicate a SavingPath.')


	Plot_PlateauList = kwargs.get('Plot_PlateauList')
	if not Plot_PlateauList:
		Plot_PlateauList = [5]
		print(f'Set Plot_PlateauList and Plot_Index to set images to plot')
	else:
		if isinstance(Plot_PlateauList, int):
			Plot_PlateauList = [Plot_PlateauList]


	Plot_Index = kwargs.get('Plot_Index')
	if not Plot_Index:
		Plot_Index = 14
		print(f'MinIndex = {MinIndex}, MinIndex-Buffer = {MinIndex-Buffer}')
		if Plot_Index>(MinIndex-Buffer):
			Plot_Index = int(MinIndex/2)
			print(f'Plot_Index outside default range. Set to {Plot_Index}, please set manually with Plot_Index')
	else:
		if Plot_Index<Buffer or Plot_Index>(MinIndex-Buffer):
			print(f'PlotIndex is outside the range of indices that will be analysed ({Buffer}, {MinIndex-Buffer})')
			Plot_Index = int(MinIndex/2)
			print(f'	Seeting it to {PlotIndex}')



	SaveHypercube = kwargs.get('SaveHypercube', True)
	if SaveHypercube:
		print(f'SaveHypercube set to {SaveHypercube}')

	print(f'Buffer set to {Buffer}')

	t0 = time.time()
	Ncolours = len(DataSweep)-1
	(_, YY, XX) = DataSweep[1].shape

	## Deal with special cases when plateau list is input as string
	if isinstance(Plot_PlateauList, str):
		if Plot_PlateauList=='All':
			Plot_PlateauList = [i for i in range(0,Ncolours)]
		elif Plot_PlateauList=='None':
			Plot_PlateauList = []


	## Define static image
	if StaticWav_index>=8:
		## to account for the fact that the dark is not included in the wavelengths list
		im_static_0 = DataSweep[StaticWav_index-1][ImStatic_Index,:,:]
	else:
		im_static_0 = DataSweep[StaticWav_index][ImStatic_Index,:,:]

	White_static_0 = WhiteHypercube[StaticWav_index_sorted,:,:]
	im_staticN_init = HySE_ManipulateHypercube.NormaliseFrames(im_static_0, White_static_0, Dark) ## HySE_ManipulateHypercube.
	im_staticN_0 = im_staticN_init

	## Loop through all colours (wavelengths)
	print(f'\n Plot_PlateauList = {Plot_PlateauList}, Plot_Index = {Plot_Index}\n')

	Hypercube = np.zeros(WhiteHypercube.shape)
	Hypercube_Masks = np.zeros(WhiteHypercube.shape)

	## Starting from static image to higher wavelengths
	for u in range(StaticWav_index_sorted, Ncolours):
		wav = Wavelengths_sorted[u]
		## Set static image for this wavelength
		im_staticN = im_staticN_0
		## Find wavelnegth index in raw data frame
		c = np.where(Wavelengths_list==wav)[0][0]
		if c>=8:
			c=c+1
		## Find white reference for wavelength
		im_white = WhiteHypercube[u,:,:]
		
		## Now co-register all frames
		ImagesTemp = []
		MasksTemp = []
		(NN, YY, XX) = DataSweep[c].shape
		for i in range(Buffer,NN-Buffer):
			im_shifted = DataSweep[c][i,:,:]
			## Masks
			mask_white = GetMask(im_white, LowCutoff=LowCutoff, HighCutoff=HighCutoff)
			mask_shifted = GetMask(im_shifted, LowCutoff=LowCutoff, HighCutoff=HighCutoff)
			mask_combined = TakeWavMaskDiff(mask_white, mask_shifted)
			MasksTemp.append(mask_combined)
			mask_combined_avg = np.average(mask_combined)
#             PlotMasksAndIm(mask_white, mask_shifted, mask_combined, im_shifted)
			
			## Normalise before co-registration
			im_shiftedN = HySE_ManipulateHypercube.NormaliseFrames(im_shifted, im_white, Dark) #HySE_ManipulateHypercube.
			
			## Co-registrate with masks
#             im_coregistered, shift_val, time_taken = CoRegisterImages(im_staticN, im_shiftedN) 
			if mask_combined_avg>mask_combined_avg_cutoff:
				## get mask for specular reflections:
				mask_shifted = GetMask(im_shifted, HighCutoff=HighCutoff)
				im_coregistered = CoRegisterImages_WithMask(im_staticN, im_shiftedN, StaticMask=mask_white, MovingMask=mask_shifted, PlotMasks=True, **kwargs)
			else:
				im_coregistered = CoRegisterImages_WithMask(im_staticN, im_shiftedN, StaticMask=mask_white, MovingMask=mask_combined, **kwargs)#, MovingMask=mask_shifted, Verbose=True)
			
			ImagesTemp.append(im_coregistered)
			## Set static image for next wavelength
			if i==ImStatic_Index:
				im_staticN_0 = im_coregistered

			## Plot co-registration if requested
			if PlotDiff:
				if c in Plot_PlateauList:
					if '.png' in SavingPath:
						NameTot = SavingPath.split('/')[-1]
						Name = NameTot.replace('.png', '')+f'_Plateau{c}_Index{i}.png'
						SavingPathWithName = SavingPath.replace(NameTot, Name)
					else:
						Name = f'_{wav}nm_Index{i}_CoRegistration.png'
						SavingPathWithName = SavingPath+Name

					if i==Plot_Index:
						HySE_UserTools.PlotCoRegistered(im_staticN, im_shiftedN, im_coregistered, SavePlot=True, SavingPathWithName=SavingPathWithName)

		ImagesTemp = np.array(ImagesTemp)
		MasksTemp = np.array(MasksTemp)
		ImAvg = np.average(ImagesTemp, axis=0)
		MasksAvg = np.round(np.average(MasksTemp, axis=0),0)
		MasksAvg = MasksAvg.astype('uint8')

		Hypercube[u,:,:] = ImAvg
		Hypercube_Masks[u,:,:] = MasksAvg


	im_staticN_0 = im_staticN_init

	## Starting now going from static wavelenght to lower wavelengths
	for uu in range(0, StaticWav_index_sorted):
		u = StaticWav_index_sorted-uu-1 ## go backwards

		wav = Wavelengths_sorted[u]
		## Set static image for this wavelength
		im_staticN = im_staticN_0
		## Find wavelnegth index in raw data frame
		c = np.where(Wavelengths_list==wav)[0][0]
		if c>=8:
			c=c+1

		## Find white reference for wavelength
		im_white = WhiteHypercube[u,:,:]
		## Now co-register all frames
		ImagesTemp = []
		(NN, YY, XX) = DataSweep[c].shape
		
		for i in range(Buffer,NN-Buffer):
			im_shifted = DataSweep[c][i,:,:]
			## Masks
			mask_white = GetMask(im_white, LowCutoff=LowCutoff, HighCutoff=HighCutoff)
			mask_shifted = GetMask(im_shifted, LowCutoff=LowCutoff, HighCutoff=HighCutoff)
			mask_combined = TakeWavMaskDiff(mask_white, mask_shifted)
			mask_combined_avg = np.average(mask_combined)
			## Normalise before co-registration
			im_shiftedN = HySE_ManipulateHypercube.NormaliseFrames(im_shifted, im_white, Dark)
			
			## Co-registrate with masks
			## Because I am doing the co-registration frame by frame and therefore am not averaging frames for a given wavelengths,
			## dark values oscillate between ~6 and exactly 0. These 0 values are caught by the mask, which can quickly add to
			## too many masked values for the elastix moving mask option. 
			## When that is the case, only consider the static mask (from the white image)
			if mask_combined_avg>mask_combined_avg_cutoff:
				# im_coregistered = CoRegisterImages_WithMask(im_staticN, im_shiftedN, StaticMask=mask_white, **kwargs)#, MovingMask=mask_shifted, Verbose=True)
				mask_shifted = GetMask(im_shifted, HighCutoff=HighCutoff)
				im_coregistered = CoRegisterImages_WithMask(im_staticN, im_shiftedN, StaticMask=mask_white, MovingMask=mask_shifted, **kwargs)#
			else:
				im_coregistered = CoRegisterImages_WithMask(im_staticN, im_shiftedN, StaticMask=mask_white, MovingMask=mask_combined, **kwargs)#
				
			ImagesTemp.append(im_coregistered)
			## Set static image for next wavelength
			if i==ImStatic_Index:
				im_staticN_0 = im_coregistered

			## Plot co-registration if requested
			if PlotDiff:
				if c in Plot_PlateauList:
					if '.png' in SavingPath:
						NameTot = SavingPath.split('/')[-1]
						Name = NameTot.replace('.png', '')+f'_Plateau{c}_Index{i}.png'
						SavingPathWithName = SavingPath.replace(NameTot, Name)
					else:
						Name = f'_{wav}nm_Index{i}_CoRegistration.png'
						SavingPathWithName = SavingPath+Name

					if i==Plot_Index:
						HySE_UserTools.PlotCoRegistered(im_staticN, im_shiftedN, im_coregistered, SavePlot=True, SavingPathWithName=SavingPathWithName)

		ImagesTemp = np.array(ImagesTemp)
		ImAvg = np.average(ImagesTemp, axis=0)
		Hypercube[u,:,:] = ImAvg

	## Calculate time taken
	tf = time.time()
	time_total = tf-t0
	minutes = int(time_total/60)
	seconds = time_total - minutes*60
	print(f'\n\n Co-registration took {minutes} min and {seconds:.0f} s in total\n')

	if SaveHypercube:
		if '.png' in SavingPath:
			NameTot = SavingPath.split('/')[-1]
			Name = NameTot.replace('.png', '')+f'_CoregisteredHypercube.npz'
			Name_wav = NameTot.replace('.png', '')+f'_CoregisteredHypercube_wavelengths.npz'
			Name_mask = NameTot.replace('.png', '')+f'_CoregisteredHypercube_masks.npz'
			SavingPathHypercube = SavingPath.replace(NameTot, Name)
			SavingPathWavelengths = SavingPath.replace(NameTot, Name_wav)
			SavingPathMasks = SavingPath.replace(NameTot, Name_mask)
		else:
			Name = f'_CoregisteredHypercube.npz'
			Name_wav = f'_CoregisteredHypercube_wavelengths.npz'
			Name_mask = f'_CoregisteredHypercube_masks.npz'
			SavingPathHypercube = SavingPath+Name
			SavingPathWavelengths = SavingPath+Name_wav
			SavingPathMasks = SavingPath+Name_mask

		np.savez(f'{SavingPathHypercube}', Hypercube)
		np.savez(f'{SavingPathWavelengths}', Wavelengths_sorted)
		np.savez(f'{SavingPathMasks}', Hypercube_Masks)

	return Hypercube, Hypercube_Masks


