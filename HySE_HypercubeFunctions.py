##
## Functions to read video from endoscope
##

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
	print(f'Available functions:')
	print(f'')
	print(f'	FindHypercube(DataPath, Wavelengths_list, **kwargs)')
	print(f'		Primary function that is called by the user to identify each wavelength in the dataset from')
	print(f'		the raw video in order to compute the hypercube')
	print(f'		Inputs:')
	print(f'			- DataPath: Path to the raw vide')
	print(f'			- kwargs: Paramters to tweak for optimal results')
	print(f'				- Help = True: to print help message')
	print(f'				- PlotGradient = True: To plot gratient of smoothed trace and detected peaks')
	print(f'					To see effect of other parameters when optimising')
	print(f'				- PrintPeaks = True: To print the list of all detected peaks and their positions')
	print(f'				- MaxPlateauSize = Integer: Set the maximal expected size for a plateau.')
	print(f'				- WindowLength = Integer: Window over which the smoothing of the trace is performed')
	print(f'					If the data consists of NxRGB cycles, this number should be a factor of 3')
	print(f'				- PolyOrder = Integer: Order of the polynomial used in smoothing (Savitzky-Golay)')
	print(f'				- PeakHeight = Float: Detection threshold applied to the gradient of the smoothed trace')
	print(f'					to find edges between neighbouring colours')
	print(f'				- PeakDistance = Integer: Minimal distance between neightbouring peaks/plateaux')
	print(f'					Depends on the repeat number, and will impact the detection of double plateaux')
	print(f'				- DarkMin = Integer: Set the minimal size of the long dark between succesive sweeps')
	print(f'					Depends on the repeat numbner, and will impact the detection of individial sweeps')
	print(f'				- PlateauSize = Integer: Set the expected average size for a plateau (in frame number)')
	print(f'					Depends on the repeat number and will impact how well double plateaux are handled')
	print(f'					Automatically adjusts expected size when plateaux are detected, but needs to be set')
	print(f'					manually if a full sweep could not be detected automatically.')
	print(f'				- CropImDimensions = [xmin, xmax, ymin, ymax]: coordinates of image crop (default Full HD)')
	print(f'				- ReturnPeaks = True: if want the list of peaks and peak distances')
	print(f'					(for manual tests, for example if fewer than 8 colours')
	print(f'				- Ncolours = integer: if different from 8 (for example, if one FSK was off)')
	print(f'		Outputs:')
	print(f'			- EdgePos: Positions indicating where each sections of frames is for each wavelength ')
	print(f'				for all sweeps in the dataset')
	print(f'')
	print(f'')
	print(f'	ComputeHypercube(DataPath, EdgePos, Wavelengths_list, **kwargs)')
	print(f'		Primary function to compute the hypercube. It inputs the path to the data and the EdgePos output from the')
	print(f'		FindHypercube function (which indicates whereto find the start for each wavelenght for each identified sweep')
	print(f'		Input:')
	print(f'			- DataPath: Path to the data')
	print(f'			- EdgePos: Position of the start of each colour for each sweep, output of FindHypercube')
	print(f'				Functions are separated to allow tweaking of parameters to properly identify ')
	print(f'				individual sweeps')	   
	print(f'			- Wavelengths_list: List of wavelengths as measured in the data (panel 4 - panel 2)')
	print(f'			- kwargs (optional): Optional parameters')
	print(f'			- BufferSize = integer : Number of frames to ignore between neighbouring colours to avoid')
	print(f'				contamination by transition frames. Might need to be adjusted for very short or very')
	print(f'				large repetitions. Default to 10')							 
	print(f'			- Name = strin')
	print(f'		Output:')
	print(f'			- Hypercube_sorted: Hypercube contained in a 3D array, with wavelengths sorted according')
	print(f'				to order_list. Shape (Nwavelengths, 1080, 1920) for HD format')					
	print(f'			- Dark: Dark average contained in 2D array')
	print(f'')
	print(f'')
	print(f'	NormaliseHypercube(Hypercube, Hypercube_White, Dark, Wavelengths_list, **kwargs)')
	print(f'			Primary function that normalises the hypercube with the white reference')
	print(f'			Input:')
	print(f'				- Hypercube : Computed from data (3D array)')
	print(f'				- Hypercube White: Computed from white reference (3D array)')
	print(f'				- Dark : Ideally Extracted from white reference (2D array)')
	print(f'				- Wavelengths_list : List of wavelengths as implemented in the data gathering (not ordered)')
	print(f'				- kwargs: optional arguments')
	print(f'					- Name: String, used for plotting and saving data')
	print(f'			Output:')
	print(f'				- Normalised Hypercube')
	print(f'')
	print(f'')
	print(f'			 ------------ Other additional functions ------------ ')
	print(f'')
	print(f'')
	print(f'	ImportData(Path, *Coords, **Info)')
	print(f'		Function to impport data.')
	print(f'		Inputs:')
	print(f'			- Coords = Nstart, Nend')
	print(f'			- Infos: default(optional)')
	print(f'				- RGB = False(True): Keep RGB format (3D size)')
	print(f'				- Trace = False(True): Calculate trace (frame avg)')
	print(f'				- CropIm = True(False): Crop the patient info')
	print(f'				- CropImDimensions = [xmin, xmax, ymin, ymax]: coordinates of image crop (default Full HD)')
	print(f'					[702,1856, 39,1039] : CCRC Full HD')
	print(f'					[263,695, 99,475] : CCRC standard/smaller canvas')
	print(f'		Outputs:')
	print(f'			- data (array)')
	print(f'')
	print(f'	ImportData_imageio(Path, *Coords, **Info)')
	print(f'		Same input/Output as ImportData, using imageio reader')
	print(f'')
	print(f'')
	print(f'	FindPeaks(trace, **kwargs)')
	print(f'		Inputs:')
	print(f'			- trace: Trace of the data (1D)')
	print(f'			- kwargs:')
	print(f'				- window_length = integer(6):(factor 3) over which the smoothing is done')
	print(f'				- polyorder = integer(1): for smoothing (<window_length)')
	print(f'				- peak_height(0.03) = float: detection threshold for plateau edge')
	print(f'				- peak_distance(14) = interger: min distance between detected edges')
	print(f'				- PlotGradient = False(True): Plot trace gradient to help set parameters for edge detection')
	print(f'		Outputs:')
	print(f'			- peaks')
	print(f'			- SGfilter (for plotting)')
	print(f'')
	print(f'')
	print(f'	GetEdgesPos(peaks_dist, DarkMin, FrameStart, FrameEnd, MaxPlateauSize, PlateauSize, printInfo=True)')
	print(f'		Function that identify sweeps from the detected peaks.')
	print(f'		Input:')
	print(f'			- Peaks_dist: outut from GetPeakDist')
	print(f'			- DarkMin: Minimal size for a long dark marking transition between successive sweeps')
	print(f'			- FrameStart: (No longer used) Allows to only consider a subsection of the dataset')
	print(f'				starting after FrameStart instead of 0')
	print(f'			- FrameEnd: (No longer used) Allows to only consider a subsectino of the dataset')
	print(f'				ending at FrameEnd instead of -1')
	print(f'			- MaxPlateauSize: Maximal expected size for a normal pleateau. Helps handle double plateau that ')
	print(f'				occur when two neightbouring colours have poor contrast and the transition')
	print(f'				cannot be detected by the peak threshold.')
	print(f'				Should be smaller than DarkMin.')
	print(f'			- PleateauSize: Expected plateay size. Helps handle double plateau that ')
	print(f'				occur when two neightbouring colours have poor contrast and the transition')
	print(f'				cannot be detected by the peak threshold.')
	print(f'			- PrintInfo: (default True): prints details about each sweep')
	print(f'')
	print(f'		Output:')
	print(f'			- EdgePos: Array containing the coordinates for each sweep and each plateau/wavelength within each sweep.')
	print(f'				Used to identify appropriate frames and then compute the hypercube.')
	print(f'			- Stats: (No longer used) Statistics about the identified sweeps. Useful for debugging.')
	print(f'')
	print(f'')
	print(f'	Rescale(im, PercMax, Crop=True')
	print(f'				Function used to crop a certain percentage of pixel values (saturated pixels for example).')
	print(f'				Sometimes handy for data visualisation.')
	print(f'				Input:')
	print(f'					- Image ')
	print(f'					- Maximal percentage (pixels at this value are set to 1)')
	print(f'					- Crop: If True, all pixels above max pixel are set to 1')
	print(f'							If False, the image is simply rescaled with pixels higher than 1')
	print(f'								(will be cropped in plotting)')
	print(f'				Output:')
	print(f'					- Rescaled image')
	print(f'')
	print(f'')
	print(f'	GetPeakDist(peaks, FrameStart, FrameEnd)')
	print(f'		Function that calculates the distance between neightbouring peaks')
	print(f'		Inputs:')
	print(f'			- peaks (output from FindPeaks')
	print(f'			- FrameStart, FrameEnd: window over which to look at distance between peaks')
	print(f'		Outputs:')
	print(f'			- peak_dist (array')
	print(f'')
	print(f'	wavelength_to_rgb(wavelength, gamma=0.8)')
	print(f'		Inputs:')
	print(f'			- wavelength: in nm')
	print(f'			- gamma: transparacy')
	print(f'		Outputs:')
	print(f'			- (r, g, b): colour values corresponding to the wavelength')
	print(f'')
	print(f'')
	return 0


def ImportData_imageio(Path, *Coords, **Info):
	"""
	Function to import the data (in full or as a trace) from the raw video
	This function uses the imageio reader
	Now favour the standard ImportData function, which uses opencv, because it appears to
	have a slightly better bit depth


	Input: 
		- Path: Full path to the video
		
		- *Coords: (optional) --> Nstart, Nend
		
			Where to start and end reading the video (in RGB frames)
			If none: function will import the full video
			If not none, expect Nstart, Nend (integer). 
			Function will read from frame Nstart until frame Nend
				
		- **Info: (optional) --> RGB, Trace
		
			- RGB = True if you want not to flatten the imported frames into 2D
					(defaul is RGB = False)
			
			- Trace = True if you want to only calculate the average of each frame
					  Can be used to identify hypercube for large datasets 
					  Will average single frames unless RGB=True, in which case it will average the whole RGB frame

			- CropIm = False if you want the full frame (image + patient info)

			- CropImDimensions = [xmin, xmax, ymin, ymax] If not using the standard dimensions for the Full HD output
								 (For example when having lower resolution data). Indicates where to crop the data to 
								 keep just the image and get rid of the patient information

	Output:
		- Array containing the data 
			shape (N_frames, Ysize, Xsize)
			
			or if RGB = True:
			shape (N_RGBframes, Ysize, Xsize, 3)
			
			or if Trace = True:
			shape (Nframes)
			
			of if Trace = True and RGB = True:
			shape (N_RGBframes)
	"""
	try:
		RGB = Info['RGB']
		print(f'Setting RGB format = {RGB}')
	except KeyError:
		RGB = False
		
	try:
		Trace = Info['Trace']
		print(f'Only importing the trace of the data')
	except KeyError:
		Trace = False

	try:
		CropIm = Info['CropIm']
	except KeyError:
		CropIm = True
	if CropIm:
		print(f'Cropping Image')
	else:
		print(f'Keeping full frame')

	try:
		CropImDimensions = Info['CropImDimensions']
		## [702,1856, 39,1039] ## xmin, xmax, ymin, ymax - CCRC SDI full canvas
		## [263,695, 99,475] ## xmin, xmax, ymin, ymax  - CCRC standard canvas
		print(f'Cropping image: x [{CropImDimensions[0]} : {CropImDimensions[1]}], \
			y [{CropImDimensions[2]}, {CropImDimensions[3]}]')
	except KeyError:
		CropImDimensions = [702,1856, 39,1039]  ## xmin, xmax, ymin, ymax  - CCRC SDI full canvas


   
	## Define start and end parameters
	vid = imageio.get_reader(Path, 'ffmpeg')
	## This reader cannot read the number of frames correctly
	## Estimate it with the duration and fps
	metadata = imageio.get_reader(Path).get_meta_data()
	NNvid = int(metadata['fps']*metadata['duration'])

	## Define the start and end frames to read
	if len(Coords)<2:
		Nstart = 0
		Nend = NNvid
		NN = NNvid
	else:
		Nstart = Coords[0]
		Nend = Coords[1]
		## Make sure that there are enough frames to read until Nend frame
		if Nend>NNvid:
			print(f'Nend = {Nend} is larger than the number of frames in the video')
			print(f'Setting Nend to the maximal number of frames for this video: {NNvid}')
			Nend = NNvid
		NN = Nend-Nstart

	## Get video parameters
	(XX, YY) = metadata['size']
	if CropIm:
		XX = CropImDimensions[1]-CropImDimensions[0]
		YY = CropImDimensions[3]-CropImDimensions[2]


	## Define function to read frames
	def GetFrames(n0, Nframes, VID, ImagePos, RGB):
		Ims = []
		if RGB:
			for n in range(n0, n0+Nframes):
				frame = VID.get_data(n)
				im = frame[ImagePos[2]:ImagePos[3], ImagePos[0]:ImagePos[1]]
				Ims.append(im)
		else:
			for n in range(n0, n0+Nframes):
				frame = VID.get_data(n)
				ima, imb, imc = ExtractImageFromFrame(frame, ImagePos) 
				Ims.append(ima)
				Ims.append(imb)
				Ims.append(imc)
		return np.array(Ims)

	## Define function read frame traces
	def GetFrameTraces(n0, Nframes, VID, ImagePos, RGB):
		Ims = []
		if RGB:
			for n in range(n0, n0+Nframes):
				frame = VID.get_data(n)
				im = frame[ImagePos[2]:ImagePos[3], ImagePos[0]:ImagePos[1]]
				Ims.append(np.average(im))
		else:
			for n in range(n0, n0+Nframes):
				frame = VID.get_data(n)
				ima, imb, imc = ExtractImageFromFrame(frame, ImagePos) 
				Ims.append(np.average(ima))
				Ims.append(np.average(imb))
				Ims.append(np.average(imc))
		return np.array(Ims)

	## Define function to extract the image from each frame
	def ExtractImageFromFrame(frame, ImagePos):
		im3D = frame[ImagePos[2]:ImagePos[3], ImagePos[0]:ImagePos[1]]
		# im3D = im3D.astype('float64')
		# im3D = im3D.astype('uint8')
		return im3D[:,:,0], im3D[:,:,1], im3D[:,:,2]
		

	## Extract the required data
	if Trace:
		data = GetFrameTraces(Nstart, Nend, vid, CropImDimensions, RGB)
	else:
		data = GetFrames(Nstart, Nend, vid, CropImDimensions, RGB)

	return data


def ImportData(Path, *Coords, **Info):
	"""
	Function to import the data (in full or as a trace) from the raw video
	Uses the opencv reader

	Input: 
		- Path: Full path to the video
		
		- *Coords: (optional) --> Nstart, Nend
		
			Where to start and end reading the video (in RGB frames)
			If none: function will import the full video
			If not none, expect Nstart, Nend (integer). 
			Function will read from frame Nstart until frame Nend
				
		- **Info: (optional) --> RGB, Trace
		
			- RGB = True if you want not to flatten the imported frames into 2D
					(defaul is RGB = False)
			
			- Trace = True if you want to only calculate the average of each frame
					  Can be used to identify hypercube for large datasets 
					  Will average single frames unless RGB=True, in which case it will average the whole RGB frame

			- CropIm = False if you want the full frame (image + patient info)

			- CropImDimensions = [xmin, xmax, ymin, ymax] If not using the standard dimensions for the Full HD output
								 (For example when having lower resolution data). Indicates where to crop the data to 
								 keep just the image and get rid of the patient information

	Output:
		- Array containing the data 
			shape (N_frames, Ysize, Xsize)
			
			or if RGB = True:
			shape (N_RGBframes, Ysize, Xsize, 3)
			
			or if Trace = True:
			shape (Nframes)
			
			of if Trace = True and RGB = True:
			shape (N_RGBframes)
	"""
	
	## Check if the data should be left in the raw RGB 3D format
	try:
		RGB = Info['RGB']
		print(f'Setting RGB format = {RGB}')
	except KeyError:
		RGB = False
		
	try:
		Trace = Info['Trace']
		print(f'Only importing the trace of the data')
	except KeyError:
		Trace = False

	try:
		CropIm = Info['CropIm']
	except KeyError:
		CropIm = True
	if CropIm:
		print(f'Cropping Image')
	else:
		print(f'Keeping full frame')

	try:
		CropImDimensions = Info['CropImDimensions']
		## [702,1856, 39,1039] ## xmin, xmax, ymin, ymax - CCRC SDI full canvas
		## [263,695, 99,475] ## xmin, xmax, ymin, ymax  - CCRC standard canvas
		print(f'Cropping image: x [{CropImDimensions[0]} : {CropImDimensions[1]}],y [{CropImDimensions[2]}, {CropImDimensions[3]}]')
	except KeyError:
		CropImDimensions = [702,1856, 39,1039]  ## xmin, xmax, ymin, ymax  - CCRC SDI full canvas



	# ## Coordinates for the image (empirical)
	# ImagePos_PCIe = [702,1856, 39,1039] ## xmin, xmax, ymin, ymax  - CCRC SDI full canvas

   
	## Define start and end parameters
	cap = cv2.VideoCapture(Path)
	NNvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if len(Coords)<2:
		Nstart = 0
		Nend = NNvid
		NN = NNvid
	else:
		Nstart = Coords[0]
		Nend = Coords[1]
		## Make sure that there are enough frames to read until Nend frame
		if Nend>NNvid:
			print(f'Nend = {Nend} is larger than the number of frames in the video')
			print(f'Setting Nend to the maximal number of frames for this video: {NNvid}')
			Nend = NNvid
		NN = Nend-Nstart

	## Get video parameters
	XX = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	YY = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	if CropIm:
		# XX = ImagePos_PCIe[1]-ImagePos_PCIe[0]
		# YY = ImagePos_PCIe[3]-ImagePos_PCIe[2]
		XX = CropImDimensions[1]-CropImDimensions[0]
		YY = CropImDimensions[3]-CropImDimensions[2]

	## Create empty array to store the data
	if RGB:
		if Trace:
			data = []
		else:
			data = np.empty((NN, YY, XX, 3), np.dtype('uint8'))
	else:
		if Trace:
			data = []
		else:
			data = np.empty((NN*3, YY, XX), np.dtype('uint8'))

	## Populate the array
	fc = 0
	ret = True
	ii = 0
	while (fc < NNvid and ret): 
		ret, frame = cap.read()
		if frame is not None:
			if CropIm:
				# print(f'fc = {fc}, frame.shape = {frame.shape}')
				frame = frame[CropImDimensions[2]:CropImDimensions[3], CropImDimensions[0]:CropImDimensions[1]]
			if (fc>=Nstart) and (fc<Nend):
				if RGB:
					if Trace:
						data.append(np.average(frame))
					else:
						data[ii] = frame
				else: ## bgr format
					if Trace:
						data.append(np.average(frame[:,:,2]))
						data.append(np.average(frame[:,:,1]))
						data.append(np.average(frame[:,:,0]))
					else:
						data[3*ii,:,:] = frame[:,:,2]
						data[3*ii+1,:,:] = frame[:,:,1]
						data[3*ii+2,:,:] = frame[:,:,0]
				ii += 1
			fc += 1
	cap.release()
	if Trace:
		data = np.array(data)
	return data



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



def FindPeaks(trace, **kwargs):
	"""
	Input: 
	
	- trace of the data (1D)
	
	- kwargs: Parameters for the smoothing of the data (savgol filter) and finding peaks
			  Sets to default numbers that typically give decent results if nothing is input
			  
			  Expects:
				window_length = integer (factor 3)
				polyorder = interger (<window_length)
				peak_height = float
				peak_distance = integer
				PlotGradient = True/False
				
	Output:
	
	- peaks
	
	- SGfilter (for plotting)
	
	
	"""
	## Check if smoothing Window Length set by user
	try:
		window_length = kwargs['WindowLength']
		## Make sure that the value is a factor of 3, if not warn user that results won't be as good
		if window_length%3!=0:
			print(f'Window length {window_length} is not a factor of 3')
			print(f'Input a factor of 3 unless you are sure that the repeat number is no longer x3 RGB frames')
		## If not, set default value
	except KeyError:
		print(f'No window length or polyorder input. Setting to 6, 1')
		window_length = 6
	
	## Check if smoothing polyorder set by user
	try:
		polyorder = kwargs['PolyOrder']
	except KeyError:
		polyorder = 1
		
	## Check if peak height set by user
	try:
		peak_height = kwargs['PeakHeight']
		print(f'Setting peak height to {peak_height}')
	except KeyError:
		peak_height = 0.03
		print(f'No peak height input, setting it to default {peak_height}')
	
	## Check if peak distance set by user
	try:
		peak_distance = kwargs['PeakDistance']
		print(f'Setting peak distance to {peak_distance}')
	except KeyError:
		peak_distance = 14
		print(f'No peak distance input, setting it to default {peak_distance}')
	
	SGfilter = savgol_filter(trace, window_length, polyorder)
	SGfilter_grad = np.abs(np.gradient(SGfilter))
	
	peaks, _ = find_peaks(SGfilter_grad, height=peak_height, distance=peak_distance)
	return peaks, SGfilter, SGfilter_grad




def GetPeakDist(peaks, FrameStart, FrameEnd):
	peaks_dist = []
	for i in range(0,len(peaks)-1):
		peaks_dist.append([peaks[i], peaks[i+1]-peaks[i]])
	return np.array(peaks_dist)





def FindHypercube(DataPath, Wavelengths_list, **kwargs):
	"""
	Input: 
	
	- DataPath: Path to the data
	
	- kwargs: Parameters for the smoothing of the data (savgol filter) and finding peaks
			  Sets to default numbers that typically give decent results if nothing is input
			  
			  See help for list of accepted kwargs
				
	Output:
	
	- EdgePos: Positions indicating where each sections of frames is for each wavelength 
			   for all sweeps in the dataset
	
	
	"""

	## Check if user wants list of optional parameters
	try:
		Help = kwargs['Help']
	except KeyError:
		Help = False
	if Help:
		print(f'List of optional parameters:')
		print(f'If none input, the code with set a default value for each.')
		print(f'	- Help = True: to print this help message')
		print(f'	- PlotGradient = True: To plot gratient of smoothed trace and detected peaks')
		print(f'			To see effect of other parameters when optimising')
		print(f'	- PrintPeaks = True: To print the list of all detected peaks and their positions')
		print(f'	- MaxPlateauSize = Integer: Set the maximal expected size for a plateau.')
		print(f'	- WindowLength = Integer: Window over which the smoothing of the trace is performed')
		print(f'			If the data consists of NxRGB cycles, this number should be a factor of 3')
		print(f'	- PolyOrder = Integer: Order of the polynomial used in smoothing (Savitzky-Golay)')
		print(f'	- PeakHeight = Float: Detection threshold applied to the gradient of the smoothed trace')
		print(f'			to find edges between neighbouring colours')
		print(f'	- PeakDistance = Integer: Minimal distance between neightbouring peaks/plateaux')
		print(f'			Depends on the repeat number, and will impact the detection of double plateaux')
		print(f'	- DarkMin = Integer: Set the minimal size of the long dark between succesive sweeps')
		print(f'			Depends on the repeat numbner, and will impact the detection of individial sweeps')
		print(f'	- PlateauSize = Integer: Set the expected average size for a plateau (in frame number)')
		print(f'			Depends on the repeat number and will impact how well double plateaux are handled')
		print(f'			Automatically adjusts expected size when plateaux are detected, but needs to be set')
		print(f'			manually if a full sweep could not be detected automatically.')
		print(f'	- CropImDimensions = [xmin, xmax, ymin, ymax]: coordinates of image crop (default Full HD)')
		print(f'	- ReturnPeaks = True: if want the list of peaks and peak distances')
		print(f'			(for manual tests, for example if fewer than 8 colours')
		print(f'	- Ncolours = integer: if different from 8 (for example, if one FSK was off)')
		return 0
	else:
		print(f'Add \'Help=True\' in input for a list and description of all optional parameters ')
	
	## Check if user wants to plot the gradient (to optimise parameters)
	try:
		PlotGradient = kwargs['PlotGradient']
	except KeyError:
		PlotGradient = False
		
	## Check if user wants to print the list of peaks and distances 
	## between them (to optimise parameters)
	try:
		PrintPeaks = kwargs['PrintPeaks']
	except KeyError:
		PrintPeaks = False
		
	## Check if user has set the max plateau size
	## Used to handle double plateaux when neighbouring wavelengths
	## give too low contrast 
	## Needs to be adjusted if chaning repeat number
	try:
		MaxPlateauSize = kwargs['MaxPlateauSize']
		print(f'Max plateau size set to {MaxPlateauSize}')
	except KeyError:
		MaxPlateauSize = 40
		print(f'Max plateay size set to default of {MaxPlateauSize}')
	  
	## Check if user has set the minimum size of long dark (separating sweeps)
	## Will vary with repeat number, should be larger than MaxPlateauSize
	try:
		DarkMin = kwargs['DarkMin']
		print(f'Min long dark size set to {DarkMin}')
	except KeyError:
		DarkMin = 90
		print(f'Min long dark size set to default of {DarkMin}')
		
	## Check if the user has input the expected plateau size
	try:
		PlateauSize = kwargs['PlateauSize']
		print(f'Expected plateau size set to {PlateauSize}')
	except KeyError:
		PlateauSize = 45
		print(f'Expected plateau size set to default {PlateauSize}')

	## Check if the user wants to return the peaks
	try:
		ReturnPeaks = kwargs['ReturnPeaks']
		print(f'ATTENTION: ReturnPeaks is set to True. Be careful, the output will have three elements!')
	except KeyError:
		ReturnPeaks = False

	## Check if the user wants to return the peaks
	try:
		Ncolours = kwargs['Ncolours']
		print(f'Assuming {Ncolours} wavelengths instead of normal 8')
	except KeyError:
		Ncolours = 8

	
	## Import trace

	## If CropImDimensions dimensions have been specified, pass on to import data function
	try:
		CropImDimensions = kwargs['CropImDimensions']
		trace = ImportData(DataPath,Trace=True, CropImDimensions=CropImDimensions)
	except KeyError: 
		trace = ImportData(DataPath,Trace=True)

	## Find peaks
	peaks, SGfilter, SGfilter_grad = FindPeaks(trace, **kwargs)
	## Find distance between peaks
	peaks_dist = GetPeakDist(peaks, 0, len(trace))
	if PrintPeaks:
		print(peaks_dist) 
	## Find sweep positions, will print edges for each identified sweep
	EdgePos, Stats = GetEdgesPos(peaks_dist, DarkMin, 0, len(trace), MaxPlateauSize, PlateauSize, Ncolours, printInfo=True)
	
	## Now make figure to make sure all is right
	SweepColors = ['royalblue', 'indianred', 'limegreen', 'gold', 'darkturquoise', 'magenta', 'orangered']
	fs = 4
	
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13,4))
	ax.plot(trace, '.-', color='gray', label='Dazzle - White')
	ax.plot(SGfilter, '-', color='black', label='Savitzky-Golay filter')
	
	if PlotGradient: ## For debugging
		## Plot gradient
		ax2=ax.twinx()
		ax2.plot(SGfilter_grad, '-', color='limegreen', label='Gradient')
		## Plot peaks
		for i in range(0,len(peaks)):
			if i==0:
				ax.axvline(peaks[i], ls='dotted', c='red', label='Plateau edge')
			else:
				ax.axvline(peaks[i], ls='solid', c='red')
		ax2.set_ylabel('Gradient', c='limegreen', fontsize=fs)
		ax2.yaxis.label.set_color('limegreen')
		
		
	for k in range(0,len(EdgePos)):
		edges = EdgePos[k]
		for i in range(0,len(edges)):
			s, ll = edges[i,0], edges[i,1]
			ax.axvline(s, ls='dotted', c=SweepColors[k])
			if i<7:
				RGB = wavelength_to_rgb(Wavelengths_list[i])
				ax.text(s+7, SGfilter[s+10]+3, Wavelengths_list[i], fontsize=fs, c=RGB)
			elif (i==7 or i==8):
				ax.text(s, SGfilter[s+10]-3, 'DARK', fontsize=fs, c='black')
			else:
				RGB = wavelength_to_rgb(Wavelengths_list[i-2])
				ax.text(s+7, SGfilter[s+10]+3, np.round(Wavelengths_list[i-2],0), fontsize=fs, c=RGB)


	# ax.legend()
	ax.set_xlabel('Frame', fontsize=16)
	ax.set_ylabel('Average image intensity', fontsize=16)

	ax.set_title('Trace and Detected Sweeps', fontsize=20)
	plt.tight_layout()
	
	## Find current path and time to save figure
	cwd = os.getcwd()
	time_now = datetime.now().strftime("%Y%m%d__%I-%M-%S-%p")
	day_now = datetime.now().strftime("%Y%m%d")
	
	plt.savefig(f'{cwd}/{time_now}_Trace.png')
	plt.show()

	if ReturnPeaks:
		return EdgePos, peaks, peaks_dist
	
	else:
		return EdgePos





def GetEdgesPos(peaks_dist, DarkMin, FrameStart, FrameEnd, MaxPlateauSize, PlateauSize, Ncolours, printInfo=True):
	"""
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
		- Ncolours: Added to handle cases when there are fewer than 8 wavelengths (one FSK off for example)
		- PrintInfo: (default True): prints details about each sweep

	Output:
		- EdgePos: Array containing the coordinates for each sweep and each plateau/wavelength within each sweep.
					Used to identify appropriate frames and then compute the hypercube.
		- Stats: (No longer used) Statistics about the identified sweeps. Useful for debugging.

	"""
	EdgePos = []
	temp = []
	Stats = []
	## Expected size of a sweep (based on number of colours (8*2) + dark (1))
	# temp_size = 17
	temp_size = 17 - (8-Ncolours)
	for i in range(0,len(peaks_dist)):
		## Check if we are within the right range to avoid errors
		if (peaks_dist[i,0]>=FrameStart and peaks_dist[i,0]<=FrameEnd):
			if peaks_dist[i,1]>DarkMin: ## Long dark - new sweep
				temp = np.array(temp)
				# print(f'  Start of a new plateau. len(temp) = {len(temp)}, time_size = {temp_size}')
				if len(temp)==temp_size:
					temp_avg = np.average(temp[:,1])
					temp_std = np.std(temp[:,1])
					if printInfo:
						print(f'\n{temp[:,1]}\n  separation: {peaks_dist[i,1]} - {len(temp)} plateaux, avg {temp_avg:.2f} frames +/- {temp_std:.2f}\n')
					EdgePos.append(temp)
					PlateauSize = int(np.round(temp_avg))
					Stats.append([temp_avg, temp_std])
				temp = []
			## If at the end of the trace, check if we have a full sweep
			elif i==(len(peaks_dist)-1):
				if len(temp)==(temp_size-1):
					x0 = peaks_dist[i,0]
					temp.append([peaks_dist[i,0], PlateauSize])                    
					temp = np.array(temp)
					temp_avg = np.average(temp[:,1])
					temp_std = np.std(temp[:,1])
					if printInfo:
						print(f'{temp[:,1]}\n  separation: {peaks_dist[i,1]} - {len(temp)} plateaux, avg {temp_avg:.2f} frames +/- {temp_std:.2f}\n')
					EdgePos.append(temp)
					Stats.append([temp_avg, temp_std])    
			else:
				## Ad hoc to fix white double plateau
				## Sometimes two neighbouring colours are too similar and the code can't pick up the difference
				# print(f'Double plateau:  peaks_dist[i, 1] = {peaks_dist[i, 1]}, MaxPlateauSize = {MaxPlateauSize}')
				if peaks_dist[i, 1]>MaxPlateauSize: 
					# print(f'Splitting plateau')
					# x0 = peaks_dist[i,0]
					# x1 = peaks_dist[i,0]+PlateauSize
					temp.append([peaks_dist[i,0], PlateauSize])
					temp.append([peaks_dist[i,0]+PlateauSize, (peaks_dist[i, 1])])
				else:
					## In the sweep, keep appending
					temp.append([peaks_dist[i,0], peaks_dist[i, 1]])
	
	## Print error message with suggestions if no sweep found
	if len(EdgePos)==0:
		print(f'\nNo sweep found. Set PlotGradient to True and play with parameters to improve detection')
		print(f'   To adjust smoothing: window_length, polyorder')
		print(f'   To adjust edge detection: peak_height, peak_distance')
		print(f'   To adjust the number of expected wavelengths: Ncolours')
	return np.array(EdgePos), np.array(Stats)



def ComputeHypercube(DataPath, EdgePos, Wavelengths_list, **kwargs):
	"""
	Function to compute the hypercube. It inputs the path to the data and
	the EdgePos output from the FindHypercube function (which indicates where
	to find the start for each wavelenght for each identified sweep)
	
	Input:
	- DataPath: Path to the data
	
	- EdgePos: Position of the start of each colour for each sweep, output of FindHypercube
			   Functions are separated to allow tweaking of parameters to properly identify 
			   individual sweeps
			   
	- Wavelengths_list: List of wavelengths as measured in the data (panel 4 - panel 2)
	
	- kwargs (optional): Optional parameters
				- BufferSize = integer : Number of frames to ignore between neighbouring 
										 colours to avoid contamination by transition frames.
										 Might need to be adjusted for very short or very large
										 repetitions.
										 Default to 10
										 
				- Name = string
										 
	
	Output:
	- Hypercube_sorted: Hypercube contained in a 3D array, with wavelengths sorted according
						to order_list
						Shape (Nwavelengths, 1080, 1920) for HD format
						
	- Dark: Dark average contained in 2D array
	
	
	"""
	## Check if the user has set the Buffer size
	try: 
		BufferSize = kwargs['BufferSize']
		print(f'Buffer of frames to ignore between neighbouring wavelenghts set to {BufferSize}')
	except KeyError:
		BufferSize = 10
		print(f'Buffer of frames to ignore between neighbouring wavelenghts set to default {BufferSize}')
	
	try: 
		Name = kwargs['Name']+'_'
	except KeyError:
		Name = ''
		
	## Import data
	data = ImportData(DataPath)
	
	## Sort Wavelengths
	order_list = np.argsort(Wavelengths_list)
	Wavelengths_sorted = Wavelengths_list[order_list]

	
	## Set parameters
	DarkN = [8] ## Indicate when to expect dark plateau when switching from panel 4 to 2
	ExpectedWavelengths = 16
	EdgeShape = EdgePos.shape
	## Check if multiple sweeps must be averaged
	if len(EdgeShape)==3:
		Nsweep, Nplateaux, _ = EdgePos.shape
	elif len(EdgeShape)==2:
		Nplateaux, _ = EdgePos.shape
		Nsweep = 1
		print('Only one sweep')
	else:
		print(f'There is a problem with the shape of EdgePos: {EdgeShape}')
	if Nplateaux!=(ExpectedWavelengths+1):
		print(f'Nplateaux = {Nplateaux} is not what is expected. Will run into problems.')
	
	## Compute Hypercube and Dark
	Hypercube = []
	Darks = []
	bs = int(np.round(BufferSize/2,0)) ## Buffer size for either side of edge
	for n in range(0,Nsweep):
		Hypercube_n = []
		for i in range(0,Nplateaux):
			if i not in DarkN: ## Skip the middle
				if Nsweep==1:
					framestart = EdgePos[i,0]
					plateau_size = EdgePos[i,1]
				else:
					framestart = EdgePos[n,i,0]
					plateau_size = EdgePos[n,i,1]
				s = framestart+bs
				e = framestart+plateau_size-bs
				## Print how many frames are averaged
				## React if number unreasonable (too small or too large)
				if i==0: 
					print(f'Computing hypercube: Averaging {e-s} frames')
				data_sub = data[s:e,:,:]
				data_avg = np.average(data_sub, axis=0)
				Hypercube_n.append(data_avg)
			else: ## Dark
				if Nsweep==1:
					framestart = EdgePos[i,0]
					plateau_size = EdgePos[i,1]
				else:
					framestart = EdgePos[n,i,0]
					plateau_size = EdgePos[n,i,1]
				s = framestart+bs
				e = framestart+plateau_size-bs
				data_sub = data[s:e,:,:]
				data_avg = np.average(data_sub, axis=0)
				Darks.append(data_avg) 
		Hypercube.append(Hypercube_n)
	Darks = np.array(Darks)
	Hypercube = np.array(Hypercube)
	## Average sweeps
	Hypercube = np.average(Hypercube,axis=0)
	Darks = np.average(Darks,axis=0)
	
	## Sort hypercube according to the order_list
	## Ensures wavelenghts are ordered from blue to red
	Hypercube_sorted = []
	for k in range(0,Hypercube.shape[0]):
		Hypercube_sorted.append(Hypercube[order_list[k]])
	Hypercube_sorted = np.array(Hypercube_sorted)
	
	## MakeFigure
	nn = 0
	Mavg = np.average(Hypercube)
	Mstd = np.std(Hypercube)
	MM = Mavg+5*Mstd
	fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
	for j in range(0,4):
		for i in range(0,4):
			if nn<17:
				wav = Wavelengths_sorted[nn]
				RGB = wavelength_to_rgb(wav)
				ax[j,i].imshow(Hypercube[nn,:,:], cmap='gray', vmin=0, vmax=MM)
				ax[j,i].set_title(f'{wav} nm', c=RGB)
				ax[j,i].set_xticks([])
				ax[j,i].set_yticks([])
				nn = nn+1
			else:
				ax[j,i].set_xticks([])
				ax[j,i].set_yticks([])
	plt.tight_layout()
	## Find current path and time to save figure
	cwd = os.getcwd()
	time_now = datetime.now().strftime("%Y%m%d__%I-%M-%S-%p")
	day_now = datetime.now().strftime("%Y%m%d")
	
	plt.savefig(f'{cwd}/{Name}{time_now}_Hypercube.png')
	np.savez(f'{cwd}/{Name}{time_now}_Hypercube.npz', Hypercube_sorted)
	np.savez(f'{cwd}/{Name}{time_now}_Dark.npz', Darks)
	return Hypercube_sorted, Darks



def NormaliseHypercube(Hypercube, Hypercube_White, Dark, Wavelengths_list, **kwargs):
	"""
	Function that normalises the hypercube with the white reference
	Input:
		- Hypercube : Computed from data (3D array)

		- Hypercube White: Computed from white reference (3D array)

		- Dark : Ideally Extracted from white reference (2D array)

		- Wavelengths_list : List of wavelengths as implemented in the data gathering (not ordered)

		- kwargs: optional arguments

			Accepts:
			- Name: String, used for plotting and saving data


	Output:
		- Normalised Hypercube


	"""
	try: 
		Name = kwargs['Name']+'_'
	except KeyError:
		Name = ''
		
	## Sort Wavelengths
	order_list = np.argsort(Wavelengths_list)
	Wavelengths_sorted = Wavelengths_list[order_list]   
		
	Nw, YY, XX = Hypercube.shape
	hypercubeN = np.zeros((Nw,YY,XX))
	## Convert to float to avoid numerical errors
	darkf = Dark.astype('float64')
	for n in range(0,Nw):
		h = Hypercube[n,:,:].astype('float64')
		w = Hypercube_White[n,:,:].astype('float64')
		## Subtract Dark
		hh = np.subtract(h,darkf)
		ww = np.subtract(w,darkf)
		## Make sure there is no negative values 
		hh = hh-np.amin(hh)
		ww = ww-np.amin(ww)
		## Divide and ignore /0
		hN = np.divide(hh,ww, out=np.zeros_like(hh), where=ww!=0)
		hypercubeN[n,:,:] = hN
#         hypercubeN[n,:,:] = hh
	
	## MakeFigurehypercubeN
	Mavg = np.average(hypercubeN)
	Mstd = np.std(hypercubeN)
	MM = Mavg+5*Mstd
	nn = 0
	fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
	for j in range(0,4):
		for i in range(0,4):
			if nn<17:
				wav = Wavelengths_sorted[nn]
				RGB = wavelength_to_rgb(wav)
				ax[j,i].imshow(hypercubeN[nn,:,:], cmap='gray', vmin=0, vmax=MM) ##vmax=np.amax(hypercubeN)
				ax[j,i].set_title(f'{wav} nm', c=RGB)
				ax[j,i].set_xticks([])
				ax[j,i].set_yticks([])
				nn = nn+1
			else:
				ax[j,i].set_xticks([])
				ax[j,i].set_yticks([])
	plt.suptitle(f'{Name} Hypercube Normalised - vmax={MM:.2f}')
	## Find current path and time to save figure
	cwd = os.getcwd()
	time_now = datetime.now().strftime("%Y%m%d__%I-%M-%S-%p")
	day_now = datetime.now().strftime("%Y%m%d")
	plt.tight_layout()
	plt.savefig(f'{cwd}/{Name}{time_now}_HypercubeNormalised.png')
	np.savez(f'{cwd}/{Name}{time_now}_HypercubeNormalised.npz', hypercubeN)
	return hypercubeN


def Rescale(im, PercMax, Crop=True):
	"""
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

	"""
	imflat = im.flatten()
	imsorted = np.sort(imflat)
	N = len(imsorted)
	Nmax = int(np.round(PercMax*N,0))
	MM = imsorted[Nmax]
	imnoneg = im-np.amin(im)
	imrescaled = imnoneg/MM
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
	ax.plot(imsorted,'.-')
	ax.axvline(Nmax,c='red')
	plt.tight_layout()
	plt.show()
	if Crop:
		pos = np.where(imrescaled>1)
		for i in range(0,len(pos[0])):
			imrescaled[pos[0][i],pos[1][i]] = 1
	return imrescaled

