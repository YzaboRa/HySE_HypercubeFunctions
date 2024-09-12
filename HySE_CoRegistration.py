
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
import SimpleITK as sitk
import time
from tqdm.notebook import trange, tqdm, tnrange


from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"


def wavelength_to_rgb(wavelength, gamma=0.8):

	'''This converts a given wavelength of light to an 
	approximate RGB color value. The wavelength must be given
	in nanometers in the range from 380 nm through 750 nm
	(789 THz through 400 THz).
	Based on code by Dan Bruton
	http://www.physics.sfasu.edu/astro/color/spectra.html
	'''

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



def SweepCoRegister_FromPath(vidPath, EdgePos, Nsweep, Wavelengths_list, **kwargs):
	## Get the data for the selected sweep
	DataSweep = GetSweepPath(vidPath, EdgePos, Nsweep, kwargs)
	## Compute the hypercube
	Hypercube = SweepCoRegister(DataSweep, Wavelengths_list, kwargs)
	return Hyeprcube
	


def GetSweepData_FromPath(vidPath, EdgePos, Nsweep, **kwargs):
	## Check if the user has specificed the image crop dimensions
	try:
		CropImDimensions = kwargs['CropImDimensions']
		CropImDimensions_input = True
	except KeyError:
		CropImDimensions_input = False

	## Import all the data
	if CropImDimensions_input:
		DataAll = HySE_HypercubeFunctions.ImportData(vidPath, CropImDimensions=CropImDimensions)
	else:
		DataAll = HySE_HypercubeFunctions.ImportData(vidPath)

	DataSweep = []
	for Nc in range(0,len(EdgePos[Nsweep])):
		Data_c = DataAll[EdgePos[Nsweep][Nc,0]:EdgePos[Nsweep][Nc,0]+EdgePos[Nsweep][Nc,1], :,:]
		DataSweep.append(Data_c)

	return DataSweep


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
	for c in tnrange(0, Ncolours):
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
								PlotCoRegistered(im_static, im_shifted, im_coregistered, SavePlot=True, SavingPathWithName=SavingPathWithName)

			
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

	return Hypercube_sorted




def FindPlottingRange(array):
	array_flat = array.flatten()
	array_sorted = np.sort(array_flat)    
	mean = np.average(array_sorted)
	std = np.std(array_sorted)
	MM = mean+3*std
	mm = mean-3*std
	return mm, MM

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
	
	kwargs: 
		- ShowPlot False(True)
		- SavePlot False(True)
		- SavingPathWithName (default '')
	
	"""
	try:
		SavingPathWithName = kwargs['SavingPathWithName']
	except KeyError:
		SavingPathWithName = ''
	
	try:
		SavePlot = kwargs['SavePlot']
	except KeyError:
		SavePlot = False
		
	try:
		ShowPlot = kwargs['ShowPlot']
	except KeyError:
		ShowPlot = False
		
	images_diff_0 = np.subtract(im_shifted.astype('float64'), im_static.astype('float64'))
	images_diff_0_avg = np.average(np.abs(images_diff_0))
#     images_diff_0_std = np.std(np.abs(images_diff_0))
	images_diff_cr = np.subtract(im_coregistered.astype('float64'), im_static.astype('float64'))
	images_diff_cr_avg = np.average(np.abs(images_diff_cr))
#     images_diff_cr_std = np.average(np.std(images_diff_cr))
	
	mmm, MMM = 0, 255
	mm0, MM0 = FindPlottingRange(images_diff_0)
	mm, MM = FindPlottingRange(images_diff_cr)
	
	norm = MidpointNormalize(vmin=mm0, vmax=MM0, midpoint=0)
	cmap = 'RdBu_r'
	
	fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,7))
	im00 = ax[0,0].imshow(im_static, cmap='gray',vmin=mmm, vmax=MMM)
	ax[0,0].set_title('Static Image')
	divider = make_axes_locatable(ax[0,0])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im00, cax=cax, orientation='vertical')

	im01 = ax[0,1].imshow(im_shifted, cmap='gray',vmin=mmm, vmax=MMM)
	ax[0,1].set_title('Shifted Image')
	divider = make_axes_locatable(ax[0,1])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im01, cax=cax, orientation='vertical')

	im02 = ax[0,2].imshow(images_diff_0, cmap=cmap, norm=norm)
	ax[0,2].set_title(f'Difference (no registration)\n avg {images_diff_0_avg:.2f}')
	divider = make_axes_locatable(ax[0,2])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im02, cax=cax, orientation='vertical')
	
	im10 = ax[1,0].imshow(im_static, cmap='gray',vmin=mmm, vmax=MMM)
	ax[1,0].set_title('Static Image')
	divider = make_axes_locatable(ax[1,0])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im10, cax=cax, orientation='vertical')

	im11 = ax[1,1].imshow(im_coregistered, cmap='gray',vmin=mmm, vmax=MMM)
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



def MakeHypercubeVideo(Hypercube, SavingPathWithName, **kwargs):
	try:
		fps = kwargs['fps']
	except KeyError:
		fps = 10


	(NN, YY, XX) = Hypercube.shape

	if '.mp4' not in SavingPathWithName:
		SavingPathWithName = SavingPathWithName+'.mp4'

	out = cv2.VideoWriter(SavingPathWithName, cv2.VideoWriter_fourcc(*'mp4v'), fps, (XX, YY), False)
	for i in range(NN):
		data = Hypercube[i,:,:].astype('uint8')
		out.write(data)
	out.release()


def PlotHypercube(Hypercube, **kwargs):
	"""
	Input
		- Hypercube (np array)
		- kwargs:
			- Wavelengths: List of sorted wavelengths (for titles colours, default black)
			- SavePlot: (default False)
			- SavingPathWithName: Where to save the plot if SavePlot=True
			- ShowPlot: (default True)
			- SameScale (default False)


	"""


	try:
		Wavelengths = kwargs['Wavelengths']
	except KeyError:
		Wavelengths = [0]
		print(f'Input \'Wavelengths\' list for better plot')

	try:
		SavePlot = kwargs['SavePlot']
	except KeyError:
		SavePlot = False
	try: 
		SavingPathWithName = kwargs['SavingPathWithName']
	except KeyError:
		SavingPathWithName = ''
		if SavePlot==True:
			print(f'SavePlot is set to True. Please input a SavingPathWithName')
	try:
		ShowPlot = kwargs['ShowPlot']
	except KeyError:
		ShowPlot = True

	try:
		SameScale = kwargs['SameScale']
	except KeyError:
		SameScale = False

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
				if SameScale:
					ax[j,i].imshow(Hypercube[nn,:,:], cmap='gray', vmin=0, vmax=np.amax(Hypercube))
				else:
					ax[j,i].imshow(Hypercube[nn,:,:], cmap='gray', vmin=0)
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
	if SavePlot:
		if '.png' not in SavingPathWithName:
			SavingPathWithName = SavingPathWithName+'_Hypercube.png'
		plt.savefig(f'{SavingPathWithName}')
	if ShowPlot:
		plt.show()
