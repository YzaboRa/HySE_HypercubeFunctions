"""

Functions to import the data (full, just a sweep, different reader)


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
import time
from tqdm.notebook import trange, tqdm, tnrange
import inspect
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"



def GetSweepData_FromPath(vidPath, EdgePos, Nsweep, **kwargs):
	"""
	Function that imports a sweep directly from the indicated path
	(Useful when the EdgePos is known already - Legacy function)

	Input:
		- vidPath
		- EdgePos
		- Nsweep
		- kwargs:
			- CropImDimensions
			- Help

	Output:
		- DataSweep

	"""
	## Check if the user has specificed the image crop dimensions

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetSweepData_FromPath))
		return 0


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


def ImportData(Path, *Coords, **kwargs):
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
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(ImportData))
		return 0

	
	## Check if the data should be left in the raw RGB 3D format
	RGB = kwargs.get('RGB', False)
	if RGB:
		print(f'Using RGB format')

	Trace = kwargs.get('Trace', False)
	if Trace:
		print(f'Only importing the trace of the data')
	
	CropIm = kwargs.get('CropIm', True)
	# if CropIm:
	# 	print(f'Cropping Image')
	# else:
	# 	print(f'Keeping full frame')

	CropImDimensions = kwargs.get('CropImDimensions')
	if not CropImDimensions:
		CropImDimensions = [702,1856, 39,1039]
		## [702,1856, 39,1039] ## xmin, xmax, ymin, ymax - CCRC SDI full canvas
		## [263,695, 99,475] ## xmin, xmax, ymin, ymax  - CCRC standard canvas
		print(f'Automatic cropping: [{CropImDimensions[0]} : {CropImDimensions[1]}],y [{CropImDimensions[2]}, {CropImDimensions[3]}]')
	else:
		print(f'Cropping image: x [{CropImDimensions[0]} : {CropImDimensions[1]}],y [{CropImDimensions[2]}, {CropImDimensions[3]}]')


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
		# print(CropImDimensions)
		if CropImDimensions[1]==-1:
			CropImDimensions[1] = XX
		if CropImDimensions[3]==-1:
			CropImDimensions[3] = YY
		# XX = ImagePos_PCIe[1]-ImagePos_PCIe[0]
		# YY = ImagePos_PCIe[3]-ImagePos_PCIe[2]
		XX = CropImDimensions[1]-CropImDimensions[0]
		YY = CropImDimensions[3]-CropImDimensions[2]
		# print(CropImDimensions)

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
				x_start = CropImDimensions[0]
				x_end = CropImDimensions[1]
				y_start = CropImDimensions[2]
				y_end = CropImDimensions[3]
				frame = frame[y_start:y_end, x_start:x_end]
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


