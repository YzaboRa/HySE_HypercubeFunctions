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
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"



def Help():
	"""
	Function to print all functions available
	"""
	info='''
	Available functions: 
	  
	 	FindHypercube(DataPath, Wavelengths_list, **kwargs) 
	 		Primary function that is called by the user to identify each wavelength in the dataset from 
	 		the raw video in order to compute the hypercube 
	 		Inputs: 
	 			- DataPath: Path to the raw vide 
	 			- kwargs: Paramters to tweak for optimal results 
	 				- Help = True: to print help message 
	 				- PlotGradient = True: To plot gratient of smoothed trace and detected peaks 
	 					To see effect of other parameters when optimising 
	 				- PrintPeaks = True: To print the list of all detected peaks and their positions 
	 				- MaxPlateauSize = Integer: Set the maximal expected size for a plateau. 
	 				- WindowLength = Integer: Window over which the smoothing of the trace is performed 
	 					If the data consists of NxRGB cycles, this number should be a factor of 3 
	 				- PolyOrder = Integer: Order of the polynomial used in smoothing (Savitzky-Golay) 
	 				- PeakHeight = Float: Detection threshold applied to the gradient of the smoothed trace 
	 					to find edges between neighbouring colours 
	 				- PeakDistance = Integer: Minimal distance between neightbouring peaks/plateaux 
	 					Depends on the repeat number, and will impact the detection of double plateaux 
	 				- DarkMin = Integer: Set the minimal size of the long dark between succesive sweeps 
	 					Depends on the repeat numbner, and will impact the detection of individial sweeps 
	 				- PlateauSize = Integer: Set the expected average size for a plateau (in frame number) 
	 					Depends on the repeat number and will impact how well double plateaux are handled 
	 					Automatically adjusts expected size when plateaux are detected, but needs to be set 
	 					manually if a full sweep could not be detected automatically. 
	 				- CropImDimensions = [xmin, xmax, ymin, ymax]: coordinates of image crop (default Full HD) 
	 				- ReturnPeaks = True: if want the list of peaks and peak distances 
	 					(for manual tests, for example if fewer than 8 colours 
	 				- Ncolours = integer: if different from 8 (for example, if one FSK was off) 
	 		Outputs: 
	 			- EdgePos: Positions indicating where each sections of frames is for each wavelength  
	 				for all sweeps in the dataset 


	 	GetCoregisteredHypercube(vidPath, EdgePos, Nsweep, Wavelengths_list, **kwargs)
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
						Total number of frames removed = 2*Buffer (default 6)
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




	 	GetDark(vidPath, EdgePos, **kwargs)
	 		Function that computes the dark from the long darks that seperate individual sweeps
			Inputs:
				- vidPath: where to find the data
				- EdgePos: Positions indicating where each sections of frames is for each wavelength  
					for all sweeps in the dataset  (output from FindHypercube)
				-kwargs: optional input
					- CropImDimensions = [xstart, xend, ystart, yend] : where to crop frames to just keep the image 
						(default values from CCRC HD video)
					- Buffer: sets the numner of frames to ignore on either side of a colour transition
						Total number of frames removed = 2*Buffer (default 6)
					- DarkRepeat: Number of extra repeat for long darks
						(default 3)
					- SaveDark: whether or not to save the dark
					- SavePath: where to save the dark

			Outputs:
				- Dark
				Saved:
					Dark



	 	PlotHypercube(Hypercube, **kwargs)
	 		Function to plot the hypercube.
			Input
				- Hypercube (np array)
				- kwargs:
					- Wavelengths: List of sorted wavelengths (for titles colours, default black)
					- SavePlot: (default False)
					- SavingPathWithName: Where to save the plot if SavePlot=True
					- ShowPlot: (default True)
					- SameScale (default False)

			Output:
				- Figure (4x4, one wavelength per subfigure)
				Saved:
				if SavePlot=True:
					Figure


	 	MakeHypercubeVideo(Hypercube, SavingPathWithName, **kwargs)
	 		Function that saves a mp4 video of the hypercube
			Input:
				- Hypercube
				- SavingPathWithName
				- kwargs:
					- fps: frame rate for the video (default 10)
			Output:
				Saved:
					mp4 video
	  
	  
	 	ComputeHypercube(DataPath, EdgePos, Wavelengths_list, **kwargs) 
	 		Primary function to compute the hypercube. It inputs the path to the data and the EdgePos output from the 
	 		FindHypercube function (which indicates whereto find the start for each wavelenght for each identified sweep 
	 		Input: 
	 			- DataPath: Path to the data 
	 			- EdgePos: Position of the start of each colour for each sweep, output of FindHypercube 
	 				Functions are separated to allow tweaking of parameters to properly identify  
	 				individual sweeps 	   
	 			- Wavelengths_list: List of wavelengths as measured in the data (panel 4 - panel 2) 
	 			- kwargs (optional): Optional parameters 
	 			- BufferSize = integer : Number of frames to ignore between neighbouring colours to avoid 
	 				contamination by transition frames. Might need to be adjusted for very short or very 
	 				large repetitions. Default to 10 							 
	 			- Name = strin 
	 		Output: 
	 			- Hypercube_sorted: Hypercube contained in a 3D array, with wavelengths sorted according 
	 				to order_list. Shape (Nwavelengths, 1080, 1920) for HD format 					
	 			- Dark: Dark average contained in 2D array 
	  
	  
	 	NormaliseHypercube(Hypercube, Hypercube_White, Dark, Wavelengths_list, **kwargs) 
	 			Primary function that normalises the hypercube with the white reference 
	 			Input: 
	 				- Hypercube : Computed from data (3D array) 
	 				- Hypercube White: Computed from white reference (3D array) 
	 				- Dark : Ideally Extracted from white reference (2D array) 
	 				- Wavelengths_list : List of wavelengths as implemented in the data gathering (not ordered) 
	 				- kwargs: optional arguments 
	 					- Name: String, used for plotting and saving data 
	 			Output: 
	 				- Normalised Hypercube 
	  
	  
	 ---------------------------- Other additional functions ----------------------------
	  
	  
	 	ImportData(Path, *Coords, **Info) 
	 		Function to impport data. 
	 		Inputs: 
	 			- Coords = Nstart, Nend 
	 			- Infos: default(optional) 
	 				- RGB = False(True): Keep RGB format (3D size) 
	 				- Trace = False(True): Calculate trace (frame avg) 
	 				- CropIm = True(False): Crop the patient info 
	 				- CropImDimensions = [xmin, xmax, ymin, ymax]: coordinates of image crop (default Full HD) 
	 					[702,1856, 39,1039] : CCRC Full HD 
	 					[263,695, 99,475] : CCRC standard/smaller canvas 
	 		Outputs: 
	 			- data (array) 
	  
	 	
	 	ImportData_imageio(Path, *Coords, **Info) 
	 		Same input/Output as ImportData, using imageio reader 
	  
	  
	 	
	 	FindPeaks(trace, **kwargs) 
	 		Inputs: 
	 			- trace: Trace of the data (1D) 
	 			- kwargs: 
	 				- window_length = integer(6):(factor 3) over which the smoothing is done 
	 				- polyorder = integer(1): for smoothing (<window_length) 
	 				- peak_height(0.03) = float: detection threshold for plateau edge 
	 				- peak_distance(14) = interger: min distance between detected edges 
	 				- PlotGradient = False(True): Plot trace gradient to help set parameters for edge detection 
	 		Outputs: 
	 			- peaks 
	 			- SGfilter (for plotting) 
	  
	  
	 	
	 	GetEdgesPos(peaks_dist, DarkMin, FrameStart, FrameEnd, MaxPlateauSize, PlateauSize, printInfo=True) 
	 		Function that identify sweeps from the detected peaks. 
	 		Input: 
	 			- Peaks_dist: outut from GetPeakDist 
	 			- DarkMin: Minimal size for a long dark marking transition between successive sweeps 
	 			- FrameStart: (No longer used) Allows to only consider a subsection of the dataset 
	 				starting after FrameStart instead of 0 
	 			- FrameEnd: (No longer used) Allows to only consider a subsectino of the dataset 
	 				ending at FrameEnd instead of -1 
	 			- MaxPlateauSize: Maximal expected size for a normal pleateau. Helps handle double plateau that  
	 				occur when two neightbouring colours have poor contrast and the transition 
	 				cannot be detected by the peak threshold. 
	 				Should be smaller than DarkMin. 
	 			- PleateauSize: Expected plateay size. Helps handle double plateau that  
	 				occur when two neightbouring colours have poor contrast and the transition 
	 				cannot be detected by the peak threshold. 
	 			- PrintInfo: (default True): prints details about each sweep 
	  
	 		Output: 
	 			- EdgePos: Array containing the coordinates for each sweep and each plateau/wavelength within each sweep. 
	 				Used to identify appropriate frames and then compute the hypercube. 
	 			- Stats: (No longer used) Statistics about the identified sweeps. Useful for debugging. 
	  
	  
	 	Rescale(im, PercMax, Crop=True 
	 				Function used to crop a certain percentage of pixel values (saturated pixels for example). 
	 				Sometimes handy for data visualisation. 
	 				Input: 
	 					- Image  
	 					- Maximal percentage (pixels at this value are set to 1) 
	 					- Crop: If True, all pixels above max pixel are set to 1 
	 							If False, the image is simply rescaled with pixels higher than 1 
	 								(will be cropped in plotting) 
	 				Output: 
	 					- Rescaled image 
	  
	  
	 	GetPeakDist(peaks, FrameStart, FrameEnd) 
	 		Function that calculates the distance between neightbouring peaks 
	 		Inputs: 
	 			- peaks (output from FindPeaks 
	 			- FrameStart, FrameEnd: window over which to look at distance between peaks 
	 		Outputs: 
	 			- peak_dist (array 
	  
	 	wavelength_to_rgb(wavelength, gamma=0.8) 
	 		Inputs: 
	 			- wavelength: in nm 
	 			- gamma: transparacy 
	 		Outputs: 
	 			- (r, g, b): colour values corresponding to the wavelength 


	'''
	
	print(f'{info}')


def FindPlottingRange(array):
	array_flat = array.flatten()
	array_sorted = np.sort(array_flat)    
	mean = np.average(array_sorted)
	std = np.std(array_sorted)
	MM = mean+3*std
	mm = mean-3*std
	return mm, MM




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




def PlotHypercube(Hypercube, **kwargs):
	"""
	Function to plot the hypercube.
	Input
		- Hypercube (np array)
		- kwargs:
			- Wavelengths: List of sorted wavelengths (for titles colours, default black)
			- SavePlot: (default False)
			- SavingPathWithName: Where to save the plot if SavePlot=True
			- ShowPlot: (default True)
			- SameScale (default False)

	Output:
		- Figure (4x4, one wavelength per subfigure)
		Saved:
		if SavePlot=True:
			Figure

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








def MakeHypercubeVideo(Hypercube, SavingPathWithName, **kwargs):
	'''
	Function that saves a mp4 video of the hypercube
	Input:
		- Hypercube
		- SavingPathWithName
		- kwargs:
			- fps: frame rate for the video (default 10)
	Output:
		Saved:
			mp4 video
	'''
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


