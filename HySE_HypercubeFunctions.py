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

		## Check if the user wants to return the peaks
	try:
		ReturnPeaks = kwargs['ReturnPeaks']
		print(f'ATTENTION: ReturnPeaks is set to True. Be careful, the output will have three elements!')
	except KeyError:
		ReturnPeaks = False

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
		if ReturnPeaks:
			return 0,0,0
		else:
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
	SweepColors = ['royalblue', 'indianred', 'limegreen', 'gold', 'darkturquoise', 'magenta', 'orangered', 'cyan', 'lime', 'hotpink']
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
				ax.axvline(peaks[i], ls='dotted', c='red')
		ax2.set_ylabel('Gradient', c='limegreen', fontsize=fs)
		ax2.yaxis.label.set_color('limegreen')
		
		
	for k in range(0,len(EdgePos)):
		edges = EdgePos[k]
		for i in range(0,len(edges)):
			s, ll = edges[i,0], edges[i,1]
			ax.axvline(s, ls='dashed', c=SweepColors[k])
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
	

	Name_withExtension = DataPath.split('/')[-1]
	Name = Name_withExtension.split('.')[0]
	Path = DataPath.replace(Name_withExtension, '')

	if PlotGradient==False:
		PathToSave = f'{Path}{time_now}_{Name}_Trace.png'
		# plt.savefig(f'{cwd}/{time_now}_Trace.png')
		print(f'Saving figure at this location: \n   {PathToSave }')
		plt.savefig(PathToSave)
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
				# print(f'At the end of trace')
				if len(temp)==(temp_size-1):
					# print(f'Have full sweep')
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
				if peaks_dist[i, 1]>MaxPlateauSize: 
					# print(f'Double plateau:  peaks_dist[i, 1] = {peaks_dist[i, 1]}, MaxPlateauSize = {MaxPlateauSize}')
					temp.append([peaks_dist[i,0], PlateauSize])
					temp.append([peaks_dist[i,0]+PlateauSize, (peaks_dist[i, 1])])
				else:
					## In the sweep, keep appending
					# print(f'  Regular sweep, keep appending')
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
	time_now = datetime.now().strftime("%Y%m%d__%I-%M-%S-%p")
	day_now = datetime.now().strftime("%Y%m%d")

	Name_withExtension = DataPath.split('/')[-1]
	Name = Name_withExtension.split('.')[0]
	Path = DataPath.replace(Name_withExtension, '')
	PathToSave = f'{Path}{time_now}_{Name}'
	
	plt.savefig(f'{PathToSave}_Hypercube.png')
	np.savez(f'{PathToSave}_Hypercube.npz', Hypercube_sorted)
	np.savez(f'{PathToSave}_Dark.npz', Darks)
	return Hypercube_sorted, Darks



def NormaliseHypercube(DataPath, Hypercube, Hypercube_White, Dark, Wavelengths_list, **kwargs):
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
	time_now = datetime.now().strftime("%Y%m%d__%I-%M-%S-%p")
	day_now = datetime.now().strftime("%Y%m%d")
	plt.tight_layout()

	Name_withExtension = DataPath.split('/')[-1]
	Name = Name_withExtension.split('.')[0]
	Path = DataPath.replace(Name_withExtension, '')
	PathToSave = f'{Path}{time_now}_{Name}'
	plt.savefig(f'{PathToSave}_HypercubeNormalised.png')
	np.savez(f'{PathToSave}_HypercubeNormalised.npz', hypercubeN)

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




####################################################################
####################################################################
####################################################################
####################################################################

import SimpleITK as sitk
import time
from tqdm.notebook import trange, tqdm, tnrange


def SweepCoRegister(DataSweep, **kwargs):
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
				- Plot_PlateauList: for which plateau(x) to plot figure. Aceepts a list of integers or "All" for all plateau (defaul "All")
				- Plot_Index: which frame (index) to plot for each selected plateau (default 14)


	Outputs:

	"""
	AllIndices = [DataSweep[i].shape[0] for i in range(0,len(DataSweep))]
	MaxIndex = np.amax(AllIndices)

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
	except KeyError:
		ImStatic_Index = 8
		if MaxIndex>ImStatic_Index:
			ImStatic_Index = int(MaxIndex/2)
			print(f'ImStatic_Index is outside default range. Set to {ImStatic_Index}, please set manually with ImStatic_Index')

	try: 
		PlotDiff = kwargs['PlotDiff']
	except KeyError:
		PlotDiff = False

	if PlotDiff:
		try: 
			SavingPath = kwargs['SavingPath']
		except KeyError:
			SavingPath = ''
			print(f'PlotDiff has been set to True. Indicate a SavingPath.')
	try: 
		Plot_PlateauList = kwargs['Plot_PlateauList']
	except:
		Plot_PlateauList = 'All'
	

	try: 
		Plot_Index = kwargs['Plot_Index']
	except:
		Plot_Index = 14
		if MaxIndex>Plot_Index:
			Plot_Index = int(MaxIndex/2)
			print(f'Plot_Index outside default range. Set to {Plot_Index}, please set manually with Plot_Index')


	print(f'Static image: plateau {ImStatic_Plateau}, index {ImStatic_Index}. Use ImStatic_Plateau and ImStatic_Index to change it.')
	print(f'Buffer set to {Buffer}')

	if Plot_PlateauList=='All' or Plot_PlateauList=='None':
		PlotPlateauString = True
	else:
		PlotPlateauString = False

	t0 = time.time()
	Ncolours = len(DataSweep)
	(_, YY, XX) = DataSweep[1].shape

	Hypercube = []

	## Define static image
	im_static = DataSweep[ImStatic_Plateau][ImStatic_Index,:,:]

	## Loop through all colours (wavelengths)
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
					if PlotPlateauString:
						if Plot_PlateauList=='All':
							if i==Plot_Index:
								Name = f'Plateau{c}_Index{i}_CoRegistration.png'
								PlotCoRegistered(im_static, im_shifted, im_coregistered, SavePlot=True, SavingPathWithName=SavingPath+Name)
					else:
						if c in Plot_PlateauList:
							if i==Plot_Index:
								Name = f'Plateau{c}_Index{i}_CoRegistration.png'
								PlotCoRegistered(im_static, im_shifted, im_coregistered, SavePlot=True, SavingPathWithName=SavingPath+Name)
			
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
	return Hypercube




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

	im12 = ax[1,2].imshow(images_diff_cr, cmap=cmap, norm=norm_shift)
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
		print(f'   Set SavingPathWithName=\'path\' to set saving path')
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


	"""


	try:
		Wavelengths = kwargs['Wavelengths']
	except KeyError:
		Wavelengths = 0
		print(f'Input \'Wavelengths\' list (sorted) for nicer plot')

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


	NN, YY, XX = Hypercube.shape

	nn = 0
	plt.close()
	fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
	for j in range(0,4):
		for i in range(0,4):
			if nn<NN:
				if Wavelengths==0:
					wav = 0
					RGB = (0,0,0) ## Set title to black if no wavelength input
				else:
					wav = Wavelengths[nn]
					RGB = wavelength_to_rgb(wav)
				ax[j,i].imshow(Hypercube[nn,:,:], cmap='gray', vmin=0, vmax=np.amax(Hypercube))
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

	
















	