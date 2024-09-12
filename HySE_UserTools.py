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