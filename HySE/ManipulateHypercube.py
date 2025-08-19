"""

Functions used to calculate and manipulate the hypercube data (compute hypercube, get the dark, normalise, etc.)


"""


import numpy as np
import cv2
import os
from datetime import datetime
from scipy.signal import savgol_filter, find_peaks
import matplotlib
from matplotlib import pyplot as plt
import imageio
import inspect
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"


import HySE.Import
import HySE.UserTools


def ComputeHypercube(DataPath, EdgePos, **kwargs):

	"""
	Function to compute the hypercube. It inputs the path to the data and
	the EdgePos output from the FindHypercube function (which indicates where
	to find the start for each wavelenght for each identified sweep)
	
	Input:
	- DataPath: Path to the data
	
	- EdgePos: Position of the start of each colour for each sweep, output of FindHypercube
			   Functions are separated to allow tweaking of parameters to properly identify 
			   individual sweeps
	
	- kwargs (optional): Optional parameters
				- Help = False 
				- Wavelengths_list: List of wavelengths as measured in the data
				- Buffer = integer : Number of frames to ignore between neighbouring 
					colours to avoid contamination by transition frames.
					Might need to be adjusted for very short or very large repetitions.
					Default to 6 
				- Name = string
				- SaveFig = True
				- SaveArray = True
				- Plot = True
				- Order = False. Set to False if doing wavelength unmixing
				- Average = True. If more than one sweep is indicated, indicates whether
					to average all sweeps before computing hypercube.
					If false, it will output as many hypercubes as sweeps.
				- ForCoRegistration = False. If True, keeps individual frames
										 
	
	Output:
	- Hypercube_sorted: Hypercube contained in a 3D array, with wavelengths sorted according
						to order_list (if Order=True)
						Shape (Nwavelengths, 1080, 1920) for HD format
						
	- Dark: Dark average contained in 2D array
	
	
	"""
	## Check if the user has set the Buffer size

	Buffer = kwargs.get('Buffer', 6)
	print(f'Buffer of frames to ignore between neighbouring wavelenghts set to 2x{Buffer}')
	BufferSize = 2*Buffer
	
	Name = kwargs.get('Name', '')
	SaveFig = kwargs.get('SaveFig', True)
	Order = kwargs.get('Order', False)

	Wavelengths_list = kwargs.get('Wavelengths_list', None)
	if Wavelengths_list:
	   ## Sort Wavelengths
	   order_list = np.argsort(Wavelengths_list)
	   Wavelengths_sorted = Wavelengths_list[order_list]
	   # print(f'Wavelengths_list: \n{Wavelengths_list}\n\n')
	   # print(f'Wavelengths_sorted: \n{Wavelengths_sorted}\n\n')
	else:
	   Order=False # Can't order by wavelengths if there's no wavelength list!

	if Order:
		print(f'Order set to True. Hypercube will be sorted according to the wavelengths list')


	SaveArray = kwargs.get('SaveArray', True)
	Help = kwargs.get('Help', False)
	if Help:
		# print(info)
		print(inspect.getdoc(ComputeHypercube))
		return 0, 0
	Plot = kwargs.get('Plot', False)
	Average = kwargs.get('Average', True)

	ForCoRegistration = kwargs.get('ForCoRegistration', False)
	if ForCoRegistration:
		print(f'Keeping individual frames for co-registration')

	CropImDimensions = kwargs.get('CropImDimensions')
	if not CropImDimensions:
		# CropImDimensions = [702,1856, 39,1039]
		# ## [702,1856, 39,1039] ## xmin, xmax, ymin, ymax - CCRC SDI full canvas
		# ## [263,695, 99,475] ## xmin, xmax, ymin, ymax  - CCRC standard canvas
		CropImDimensions = [663,1818,9,1013]
		## [702,1856, 39,1039] ## xmin, xmax, ymin, ymax - CCRC SDI full canvas
		## [263,695, 99,475] ## xmin, xmax, ymin, ymax  - CCRC standard canvas
		## [663,1818,9,1013] ## xmin, xmax, ymin, ymax  - CCRC standard canvas since August 2025
		print(f'Automatic cropping: [{CropImDimensions[0]} : {CropImDimensions[1]}],y [{CropImDimensions[2]}, {CropImDimensions[3]}]')
	else:
		print(f'Cropping image: x [{CropImDimensions[0]} : {CropImDimensions[1]}],y [{CropImDimensions[2]}, {CropImDimensions[3]}]')

		
	## Import data
	data = HySE.Import.ImportData(DataPath, CropImDimensions=CropImDimensions)

	
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
				# if Nsweep==1:
				# 	framestart = EdgePos[i,0]
				# 	plateau_size = EdgePos[i,1]
				# else:
				framestart = EdgePos[n,i,0]
				plateau_size = EdgePos[n,i,1]

				s = framestart+bs
				e = framestart+plateau_size-bs
				# print(f's: {s}, e: {e}')
				s = int(s)
				e = int(e)
				## Print how many frames are averaged
				## React if number unreasonable (too small or too large)
				if i==0: 
					if ForCoRegistration==False:
						print(f'Computing hypercube: Averaging {e-s} frames')
				data_sub = data[s:e,:,:]
				if ForCoRegistration:
					data_avg = data_sub
				else:
					data_avg = np.average(data_sub, axis=0)
				Hypercube_n.append(data_avg)
			else: ## Dark
				# if Nsweep==1:
				# 	framestart = EdgePos[i,0]
				# 	plateau_size = EdgePos[i,1]
				# else:
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
	if Average:
		Hypercube = np.average(Hypercube,axis=0)
	## Always average darks
	Darks = np.average(Darks,axis=0)
	
	# Sort hypercube according to the order_list
	# Ensures wavelenghts are ordered from blue to red
	if Order:
		if ForCoRegistration:
			print(f'Ordering hypercube when keeping all frames (ForCoRegistration = True) is not supported.')
		if Average:
			Hypercube_sorted = []
			for k in range(0,Hypercube.shape[0]):
				Hypercube_sorted.append(Hypercube[order_list[k]])
			Hypercube_sorted = np.array(Hypercube_sorted)
		else:
			print(f'Warning: in HySE.ManipulateHypercube.ComputeHypercube(), Order=True but Average=False.')
			Hypercube_sorted = []
			NN, WW, YY, XX = Hypercube.shape
			for n in range(0,NN):
				hypercube_sorted_sub = []
				for k in range(0,WW):
					hypercube_sorted_sub.append(Hypercube[n,order_list[k],:,:])
				Hypercube_sorted.append(hypercube_sorted_sub)
			Hypercube_sorted = np.array(Hypercube_sorted)

	else:
		Hypercube_sorted = Hypercube

	# Hypercube_sorted = Hypercube
	# print(f'order_list: \n{order_list}')

	## Find current path and time for saving
	time_now = datetime.now().strftime("%Y%m%d__%I-%M-%S-%p")
	day_now = datetime.now().strftime("%Y%m%d")

	Name_withExtension = DataPath.split('/')[-1]
	Name = Name_withExtension.split('.')[0]
	Path = DataPath.replace(Name_withExtension, '')
	
	## MakeFigure
	if Plot:
		if Average==False:
			HypercubeToPlot = Hypercube_sorted[0,:,:,:]
			print(f'Plotting hypercube for sweep 0')
		else: 
			HypercubeToPlot = Hypercube_sorted
		nn = 0
		Mavg = np.average(HypercubeToPlot)
		Mstd = np.std(HypercubeToPlot)
		MM = Mavg+5*Mstd
		fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
		for j in range(0,4):
			for i in range(0,4):
				if nn<17:
					wav = Wavelengths_sorted[nn]
					RGB = HySE.UserTools.wavelength_to_rgb(wav)
					if ForCoRegistration:
						ax[j,i].imshow(HypercubeToPlot[nn,0,:,:], cmap='gray')
					else:
						ax[j,i].imshow(HypercubeToPlot[nn,:,:], cmap='gray')
					if Order:
						ax[j,i].set_title(f'{wav} nm', c=RGB)
					else:
						ax[j,i].set_title(f'im {nn}')
					ax[j,i].set_xticks([])
					ax[j,i].set_yticks([])
					nn = nn+1
				else:
					ax[j,i].set_xticks([])
					ax[j,i].set_yticks([])
		plt.tight_layout()
		if SaveFig:
			PathToSave = f'{Path}{time_now}_{Name}'
			plt.savefig(f'{PathToSave}_Hypercube.png')
			print(f'Saved figure at {PathToSave}')

	if SaveArray:
		np.savez(f'{PathToSave}_Hypercube.npz', Hypercube_sorted)
		np.savez(f'{PathToSave}_AutoDark.npz', Darks)
	return Hypercube_sorted, Darks



def ComputeHypercube_RGB(DataPath, EdgePos, **kwargs):
	"""
	Function to compute the hypercube. It inputs the path to the data and
	the EdgePos output from the FindHypercube function (which indicates where
	to find the start for each wavelenght for each identified sweep)
	
	Input:
	- DataPath: Path to the data
	
	- EdgePos: Position of the start of each colour for each sweep, output of FindHypercube
			   Functions are separated to allow tweaking of parameters to properly identify 
			   individual sweeps
	
	- kwargs (optional): Optional parameters
				- Help = False 
				- Wavelengths_list: List of wavelengths as measured in the data
				- Buffer = integer : Number of frames to ignore between neighbouring 
					colours to avoid contamination by transition frames.
					Might need to be adjusted for very short or very large repetitions.
					Default to 6 
				- Name = string
				- SaveFig = False
				- SaveArray = False
				- Plot = True
				- Order = False. Set to False if doing wavelength unmixing
				- Average = False. If more than one sweep is indicated, indicates whether
					to average all sweeps before computing hypercube.
					If false, it will output as many hypercubes as sweeps.
				- ForCoRegistration = True. If True, keeps individual frames
										 
	
	Output:
	- Hypercube_sorted: Hypercube contained in a 3D array, with wavelengths sorted according
						to order_list (if Order=True)
						Shape (Nwavelengths, 1080, 1920) for HD format
						
	- Dark: Dark average contained in 3D array (BGR)
	
	
	"""

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(ComputeHypercube))
		return 0, 0


	## Check if the user has set the Buffer size

	Buffer = kwargs.get('Buffer', 6)
	print(f'Buffer of frames to ignore between neighbouring wavelenghts set to 2x{Buffer}')
	BufferSize = 2*Buffer
	
	Name = kwargs.get('Name', '')
	SaveFig = kwargs.get('SaveFig', False)
	Order = kwargs.get('Order', False)

	Wavelengths_list = kwargs.get('Wavelengths_list', None)
	if Wavelengths_list:
	   ## Sort Wavelengths
	   order_list = np.argsort(Wavelengths_list)
	   Wavelengths_sorted = Wavelengths_list[order_list]
	   # print(f'Wavelengths_list: \n{Wavelengths_list}\n\n')
	   # print(f'Wavelengths_sorted: \n{Wavelengths_sorted}\n\n')
	else:
	   Order=False # Can't order by wavelengths if there's no wavelength list!

	if Order:
		print(f'Order set to True. Hypercube will be sorted according to the wavelengths list')


	SaveArray = kwargs.get('SaveArray', False)
	
	Plot = kwargs.get('Plot', False)
	Average = kwargs.get('Average', False)

	ForCoRegistration = kwargs.get('ForCoRegistration', True)
	if ForCoRegistration:
		print(f'Keeping individual frames for co-registration')

	CropImDimensions = kwargs.get('CropImDimensions')
	if not CropImDimensions:
		# CropImDimensions = [702,1856, 39,1039]
		# ## [702,1856, 39,1039] ## xmin, xmax, ymin, ymax - CCRC SDI full canvas
		# ## [263,695, 99,475] ## xmin, xmax, ymin, ymax  - CCRC standard canvas
		CropImDimensions = [663,1818,9,1013]
		## [702,1856, 39,1039] ## xmin, xmax, ymin, ymax - CCRC SDI full canvas
		## [263,695, 99,475] ## xmin, xmax, ymin, ymax  - CCRC standard canvas
		## [663,1818,9,1013] ## xmin, xmax, ymin, ymax  - CCRC standard canvas since August 2025
		print(f'Automatic cropping: [{CropImDimensions[0]} : {CropImDimensions[1]}],y [{CropImDimensions[2]}, {CropImDimensions[3]}]')
	else:
		print(f'Cropping image: x [{CropImDimensions[0]} : {CropImDimensions[1]}],y [{CropImDimensions[2]}, {CropImDimensions[3]}]')

		
	## Import data
	data = HySE.Import.ImportData(DataPath, CropImDimensions=CropImDimensions, RGB=True)

	
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
				# if Nsweep==1:
				# 	framestart = EdgePos[i,0]
				# 	plateau_size = EdgePos[i,1]
				# else:
				framestart = EdgePos[n,i,0]
				plateau_size = EdgePos[n,i,1]

				s = framestart+bs
				e = framestart+plateau_size-bs
				# print(f's: {s}, e: {e}')
				s = int(s)
				e = int(e)
				## Print how many frames are averaged
				## React if number unreasonable (too small or too large)
				if i==0: 
					if ForCoRegistration==False:
						print(f'Computing hypercube: Averaging {e-s} frames')
				data_sub = data[s:e,:,:,:]
				if ForCoRegistration:
					data_avg = data_sub
				else:
					data_avg = np.average(data_sub, axis=0)
				Hypercube_n.append(data_avg)
			else: ## Dark
				# if Nsweep==1:
				# 	framestart = EdgePos[i,0]
				# 	plateau_size = EdgePos[i,1]
				# else:
				framestart = EdgePos[n,i,0]
				plateau_size = EdgePos[n,i,1]

				s = framestart+bs
				e = framestart+plateau_size-bs
				data_sub = data[s:e,:,:,:]
				data_avg = np.average(data_sub, axis=0)
				Darks.append(data_avg) 
		Hypercube.append(Hypercube_n)
	Darks = np.array(Darks)
	Hypercube = np.array(Hypercube)
	## Average sweeps
	if Average:
		Hypercube = np.average(Hypercube,axis=0)
	## Always average darks
	Darks = np.average(Darks,axis=0)
	
	# Sort hypercube according to the order_list
	# Ensures wavelenghts are ordered from blue to red
	if Order:
		if ForCoRegistration:
			print(f'Ordering hypercube when keeping all frames (ForCoRegistration = True) is not supported.')
		if Average:
			Hypercube_sorted = []
			for k in range(0,Hypercube.shape[0]):
				Hypercube_sorted.append(Hypercube[order_list[k]])
			Hypercube_sorted = np.array(Hypercube_sorted)
		else:
			print(f'Warning: in HySE.ManipulateHypercube.ComputeHypercube(), Order=True but Average=False.')
			Hypercube_sorted = []
			NN, WW, YY, XX = Hypercube.shape
			for n in range(0,NN):
				hypercube_sorted_sub = []
				for k in range(0,WW):
					hypercube_sorted_sub.append(Hypercube[n,order_list[k],:,:,:])
				Hypercube_sorted.append(hypercube_sorted_sub)
			Hypercube_sorted = np.array(Hypercube_sorted)

	else:
		Hypercube_sorted = Hypercube

	# Hypercube_sorted = Hypercube
	# print(f'order_list: \n{order_list}')

	## Find current path and time for saving
	time_now = datetime.now().strftime("%Y%m%d__%I-%M-%S-%p")
	day_now = datetime.now().strftime("%Y%m%d")

	Name_withExtension = DataPath.split('/')[-1]
	Name = Name_withExtension.split('.')[0]
	Path = DataPath.replace(Name_withExtension, '')
	
	## MakeFigure
	if Plot:
		if Average==False:
			HypercubeToPlot = Hypercube_sorted[0,:,:,:,1]
			print(f'Plotting hypercube for sweep 0, green frame')
		else: 
			HypercubeToPlot = Hypercube_sorted[:,:,1]
			print(f'Plotting hypercube for green frame')
		nn = 0
		Mavg = np.average(HypercubeToPlot)
		Mstd = np.std(HypercubeToPlot)
		MM = Mavg+5*Mstd
		fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
		for j in range(0,4):
			for i in range(0,4):
				if nn<17:
					wav = Wavelengths_sorted[nn]
					RGB = HySE.UserTools.wavelength_to_rgb(wav)
					if ForCoRegistration:
						ax[j,i].imshow(HypercubeToPlot[nn,0,:,:], cmap='gray')
					else:
						ax[j,i].imshow(HypercubeToPlot[nn,:,:], cmap='gray')
					if Order:
						ax[j,i].set_title(f'{wav} nm', c=RGB)
					else:
						ax[j,i].set_title(f'im {nn}')
					ax[j,i].set_xticks([])
					ax[j,i].set_yticks([])
					nn = nn+1
				else:
					ax[j,i].set_xticks([])
					ax[j,i].set_yticks([])
		plt.tight_layout()
		if SaveFig:
			PathToSave = f'{Path}{time_now}_{Name}'
			plt.savefig(f'{PathToSave}_Hypercube.png')
			print(f'Saved figure at {PathToSave}')

	if SaveArray:
		PathToSave = f'{Path}{time_now}_{Name}'
		np.savez(f'{PathToSave}_Hypercube.npz', Hypercube_sorted)
		np.savez(f'{PathToSave}_AutoDark.npz', Darks)
	return Hypercube_sorted, Darks



def GetLongDark(vidPath, EdgePos, **kwargs):
	"""
	Computes dark frame from the long darks between sweeps. 
	Requires at least 2 sweeps idenfitied

	Input:
		- vidPath: Path to data
		- EdgePos: Sweep positions

		- kwargs:
			- Help: Print this help information
			- ExtraWav = 0: If an extra plateau exists at the end
			(for example when adding a red extra wavelength to mark the end of a sweep)
			- Buffer = 20: Number of frames to remove at the start and end of the sweep

	Ouput:
		- LongDark: 2D numpy array
	"""
	Help = kwargs.get('Help', False)
	if Help:
		# print(info)
		print(inspect.getdoc(GetLongDark))
		return 0

	ExtraWav = kwargs.get('ExtraWav', 0)
	Buffer = kwargs.get('Buffer', 20)


	(Nsweeps, Nw, _) = EdgePos.shape
	DataAll = HySE.Import.ImportData(vidPath) # DataAll = Import.ImportData(vidPath, **kwargs)
	Ndarks = Nsweeps-1
	## Automatic dark checks parameters
	std_Max = 5
	avg_Max = 25
	Darks = []
#     print(f'Adding {ExtraWav*EdgePos[0,-1,1]}')
	for n in range(0,Ndarks):
		print(EdgePos[n,-1,0], ExtraWav, EdgePos[n,-1,1], Buffer)
		sweep_end = EdgePos[n,-1,0] + (ExtraWav+1)*EdgePos[n,-1,1]+Buffer
		sweep_start = EdgePos[n+1,0,0]-Buffer
		if (sweep_start-sweep_end)<0:
			print(f'Not enough frames to calculate the long dark. Check EdgePos and Buffer values')
			# print(f'sweep_n_end: {sweep_end}, sweep_n+1_start: {sweep_start}')
			return 0
		else:
			print(f'Averaging {sweep_start-sweep_end} frames')
		frames = DataAll[sweep_end:sweep_start]
		## Sanity check to make sure we are only keeping dark frames
		m, M = np.amin(frames), np.amax(frames)
		avg, std = np.average(frames), np.std(frames)
		if (std>std_Max or avg>avg_Max):
			print(f'It seems like there are outlier parameters in the dark frames')
			print(f'   min = {m:.2f}, max = {M:.2f}, avg = {avg:.2f}, std = {std:.2f}')
			print(f'Start: {sweep_end}, End: {sweep_start}')
#             print(f'   Use \'DarkRepeat\' and \'Buffer\' to adjust the dark selection')
		dark = np.average(frames,axis=0)
		Darks.append(dark)
		
	Darks = np.array(Darks)
	DarkAvg = np.average(Darks,axis=0)
	return DarkAvg


def GetDark_WholeVideo(vidPath, **kwargs):
	"""
	Computes dark frame from a dark video

	Input:
		- vidPath: Path to data

		- kwargs:
			- Help: Print this help information
			- CropImDimensions: Cropping coordinates to extract endoscopy image from raw frame
				Form: [xmin, xmax, ymin, ymax] 

	Ouput:
		- LongDark: 2D numpy array
	"""
	Help = kwargs.get('Help', False)
	if Help:
		# print(info)
		print(inspect.getdoc(GetLongDark))
		return 0

	CropImDimensions = kwargs.get('CropImDimensions')
	if CropImDimensions is None:
		DataAll = HySE.Import.ImportData(vidPath)
	else:
		DataAll = HySE.Import.ImportData(vidPath, CropImDimensions=CropImDimensions)
	 # DataAll = Import.ImportData(vidPath, **kwargs)

	L = len(DataAll)
	print(f'Dataset size:  {L}')
	if L>300:
		chunk_size = 100 ## How many frames to average
	else:
		chunk_size = 10
	N = int(L/chunk_size)
	print(f'There are {N} chunks of {chunk_size} frames in this dataset')

	Darks = []
	STDs = []

	for n in range(0,N):
		start = n*chunk_size
		end = (n+1)*chunk_size
		frames = DataAll[start:end]
		avg, std = np.average(frames, axis=0), np.std(frames)
		# print(f'  dark {n}: [{start}:{end}] - {std:.2f} ')
		Darks.append(avg)
		STDs.append(std)

	dark = np.average(np.array(Darks), axis=0)
	std_est = np.average(STDs)
	print(f'Dark from video ({len(DataAll)} frames -> {chunk_size*N}). Average value: {np.average(dark):.2f}, std: {std_est:.2f}')
	dark = np.average(frames,axis=0)
	return dark


def NormaliseFrames(image, image_white, image_dark):
	"""
	Normalises an image with white and dark references
	Returns (im-dark)/(white-dark)

	Inputs:
	- image
	- image_white
	- image_dark

	Returns:
	- image_normalised

	"""
	## Convert to float to avoid numerical errors
	im = image.astype('float64')
	white = image_white.astype('float64')
	dark = image_dark.astype('float64')
	# Subtract dark
	im_d = np.subtract(im, dark)
	white_d = np.subtract(white, dark)
	## avoid negative
	## NB: the value subtracted rounds up to 0, but avoids
	## running into huge numerical errors when dividing
	im_d = im_d - np.amin(im_d)
	white_d = white_d - np.amin(white_d)
	## Divide image by white, avoiding /0 errors
	im_n = np.divide(im_d, white_d, out=np.zeros_like(im_d), where=white_d!=0)

	return im_n


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


def ComputeSelfNormHypercube(Video,EdgePos,DarkRef=None,**kwargs):
	"""
	Function to compute the hypercube. It inputs the path to the data and
	the EdgePos output from the FindHypercube function (which indicates where
	to find the start for each wavelenght for each identified sweep)

	:param Video:
	:param EdgePos:
	:return:

	Input:
	- Video: Video already imported and cropped by ImportData function

	- EdgePos: Position of the start of each colour for each sweep, output of FindHypercube
			   Functions are separated to allow tweaking of parameters to properly identify
			   individual sweeps

	- kwargs (optional): Optional parameters
				- Help = False
				- Buffer = integer : Number of frames to ignore between neighbouring
					colours to avoid contamination by transition frames.
					Might need to be adjusted for very short or very large repetitions.
					Default to 6
				- SaveFig = False
				- SaveArray = False
				- Plot = True
				# - Order = True. Set to False if doing wavelength unmixing
				- Average = True. If more than one sweep is indicated, indicates whether
					to average all sweeps before computing hypercube.
					If false, it will output as many hypercubes as sweeps.
				- ComputeDarks = False. If True, compute the average Dark frames too
				- RefChannel = None. Index of reference channel. If None, average is taken across all channels.
				- Normalise = False. If True, normalise to reference channel. If false, omit reference channel

	Output:
	- Hypercube_sorted: Hypercube contained in a 3D array, with wavelengths sorted according
						to order_list (if Order=True)
						Shape (Nwavelengths, 1080, 1920) for HD format

	- Dark: Dark average contained in 2D array


	"""
	## Check if the user has set the Buffer size
	RGB = kwargs.get('RGB', True)
	Buffer = kwargs.get('Buffer', 2)
	print(f'Buffer of frames to ignore between neighbouring wavelengths set to 2x{Buffer}')
	BufferSize = 2 * Buffer

	RefChannel = kwargs.get('RefChannel', None)
	if RefChannel not in [None,0,1,2]:
		print(f'RefChannel {RefChannel} not valid. Must be in [None,0,1,2]')
	Normalise = kwargs.get('Normalise', False)

	ComputeDarks = kwargs.get('ComputeDarks', False)
	if ComputeDarks:
		Darks = []

	SaveFig = kwargs.get('SaveFig', False)
	SaveArray = kwargs.get('SaveArray', False)
	PathToSave = kwargs.get('PathToSave')

	Help = kwargs.get('Help', False)
	if Help:
		# print(info)
		print(inspect.getdoc(ComputeHypercube))
		return 0, 0
	Plot = kwargs.get('Plot', False)
	Average = kwargs.get('Average', True)

	if np.array(DarkRef).all() == None:
		DarkRef = np.zeros(shape=Video.shape[1:4])

	## Set parameters
	DarkN = [8]  ## Indicate when to expect dark plateau when switching from panel 4 to 2
	ExpectedWavelengths = 16
	EdgeShape = EdgePos.shape
	## Check if multiple sweeps must be averaged
	if len(EdgeShape) == 3:
		Nsweep, Nplateaux, _ = EdgePos.shape
	elif len(EdgeShape) == 2:
		Nplateaux, _ = EdgePos.shape
		Nsweep = 1
		print('Only one sweep')
	else:
		print(f'There is a problem with the shape of EdgePos: {EdgeShape}')
	if Nplateaux != (ExpectedWavelengths + 1):
		print(f'Nplateaux = {Nplateaux} is not what is expected. Will run into problems.')

	## Compute Hypercube and Dark
	Hypercube = []
	bs = int(np.round(BufferSize / 2, 0))  ## Buffer size for either side of edge
	for n in range(0, Nsweep):
		Hypercube_n = []
		for i in range(0, Nplateaux):
			if i not in DarkN:  ## Skip the middle
				# if Nsweep==1:
				# 	framestart = EdgePos[i,0]
				# 	plateau_size = EdgePos[i,1]
				# else:
				framestart = EdgePos[n, i, 0]
				plateau_size = EdgePos[n, i, 1]

				s = framestart + bs
				e = framestart + plateau_size - bs
				# print(f's: {s}, e: {e}')
				s = int(s)
				e = int(e)
				## Print how many frames are averaged
				## React if number unreasonable (too small or too large)
				if i == 0:
					print(f'Computing hypercube: Averaging {e - s} frames')
				# Convert video section to float32 before normalising to avoid numerical errors
				data_norm = Video[s:e].astype('float32')-DarkRef[None,:,:,:]
				data_norm[data_norm<=0]=0 # Remove negative values after dark subtraction
				if RefChannel:
					if Normalise:
						data_norm = np.divide(np.delete(data_norm,RefChannel,axis=3).mean(axis=3),data_norm[:,:,:,RefChannel],
											  out=np.zeros(shape=np.shape(data_norm)[0:3]),where=data_norm[:, :, :, RefChannel]!=0,
											  dtype='float32')
					else:
						data_norm = np.delete(data_norm,RefChannel,axis=3).mean(axis=3,dtype='float32')
				else:
					data_norm = data_norm.mean(axis=3,dtype='float32')
				data_avg = np.mean(data_norm, axis=0,dtype='float32')
				Hypercube_n.append(data_avg)
			else:  ## Dark
				if ComputeDarks:
					framestart = EdgePos[n, i, 0]
					plateau_size = EdgePos[n, i, 1]
					s = framestart + bs
					e = framestart + plateau_size - bs
					data_sub = Video[s:e, :, :]
					data_avg = np.mean(data_sub, axis=0)
					Darks.append(data_avg)
		Hypercube.append(Hypercube_n)
	if ComputeDarks:
		Darks = np.array(Darks)
	Hypercube = np.array(Hypercube)
	## Average sweeps
	if Average:
		Hypercube = np.mean(Hypercube, axis=0)
	## Always average darks
	if ComputeDarks:
		Darks = np.mean(Darks, axis=0)

	# Hypercube_sorted = Hypercube
	# print(f'order_list: \n{order_list}')

	## Find current path and time for saving
	time_now = datetime.now().strftime("%Y%m%d__%I-%M-%S-%p")
	day_now = datetime.now().strftime("%Y%m%d")

	## MakeFigure
	if Plot:
		if Average == False:
			HypercubeToPlot = Hypercube[0, :, :, :]
			print(f'Plotting hypercube for sweep 0')
		else:
			HypercubeToPlot = Hypercube
		nn = 0
		Mavg = np.mean(HypercubeToPlot)
		Mstd = np.std(HypercubeToPlot)
		MM = Mavg + 5 * Mstd
		fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
		for j in range(0, 4):
			for i in range(0, 4):
				if nn < 17:
					ax[j, i].imshow(HypercubeToPlot[nn, :, :], cmap='gray')
					ax[j, i].set_title(f'im {nn}')
					ax[j, i].set_xticks([])
					ax[j, i].set_yticks([])
					nn = nn + 1
				else:
					ax[j, i].set_xticks([])
					ax[j, i].set_yticks([])
		plt.tight_layout()
		if PathToSave:
			plt.savefig(f'{PathToSave}_Hypercube.png')

	if SaveArray:
		if PathToSave:
			np.savez(f'{PathToSave}_Hypercube.npz', Hypercube)
			if ComputeDarks:
				np.savez(f'{PathToSave}_AutoDark.npz', Darks)
		else:
			print("Can't save data without PathToSave")
	if ComputeDarks:
		return Hypercube, Darks
	else:
		return Hypercube

def ComputeSelfNormHypercubeFromFile(DataPath,EdgePos,DarkRef=None,**kwargs):
	"""
	Function to compute the hypercube. It inputs the path to the data and
	the EdgePos output from the FindHypercube function (which indicates where
	to find the start for each wavelenght for each identified sweep)

	:param Video:
	:param EdgePos:
	:param DarkRef:
	:return:

	Input:
	- Video: Video already imported and cropped by ImportData function

	- EdgePos: Position of the start of each colour for each sweep, output of FindHypercube
			   Functions are separated to allow tweaking of parameters to properly identify
			   individual sweeps

	- kwargs (optional): Optional parameters
				- Help = False
				- Buffer = integer : Number of frames to ignore between neighbouring
					colours to avoid contamination by transition frames.
					Might need to be adjusted for very short or very large repetitions.
					Default to 6
				- SaveFig = False
				- SaveArray = False
				- Plot = True
				# - Order = True. Set to False if doing wavelength unmixing
				- Average = True. If more than one sweep is indicated, indicates whether
					to average all sweeps before computing hypercube.
					If false, it will output as many hypercubes as sweeps.
				- ComputeDarks = False. If True, compute the average Dark frames too
				- RefChannel = None. Index of reference channel. If None, average is taken across all channels.
				- Normalise = False. If True, normalise to reference channel. If false, omit reference channel

	Output:
	- Hypercube_sorted: Hypercube contained in a 3D array, with wavelengths sorted according
						to order_list (if Order=True)
						Shape (Nwavelengths, 1080, 1920) for HD format

	- Dark: Dark average contained in 2D array


	"""
	## Check if the user has set the Buffer size
	RGB = kwargs.get('RGB', True)
	Buffer = kwargs.get('Buffer', 2)
	print(f'Buffer of frames to ignore between neighbouring wavelengths set to 2x{Buffer}')
	BufferSize = 2 * Buffer

	RefChannel = kwargs.get('RefChannel', None)
	if RefChannel not in [None,0,1,2]:
		print(f'RefChannel {RefChannel} not valid. Must be in [None,0,1,2]')
	Normalise = kwargs.get('Normalise', False)

	ComputeDarks = kwargs.get('ComputeDarks', False)
	if ComputeDarks:
		Darks = []

	SaveFig = kwargs.get('SaveFig', False)
	SaveArray = kwargs.get('SaveArray', False)
	PathToSave = kwargs.get('PathToSave')

	Help = kwargs.get('Help', False)
	if Help:
		# print(info)
		print(inspect.getdoc(ComputeHypercube))
		return 0, 0
	Plot = kwargs.get('Plot', False)
	Average = kwargs.get('Average', True)

	CropImDimensions = kwargs.get('CropImDimensions')
	if not CropImDimensions:
		CropImDimensions = [702, 1856, 39, 1039]
		## [702,1856, 39,1039] ## xmin, xmax, ymin, ymax - CCRC SDI full canvas
		## [263,695, 99,475] ## xmin, xmax, ymin, ymax  - CCRC standard canvas
		print(
			f'Automatic cropping: [{CropImDimensions[0]} : {CropImDimensions[1]}],y [{CropImDimensions[2]}, {CropImDimensions[3]}]')
	else:
		print(f'Cropping image: x [{CropImDimensions[0]} : {CropImDimensions[1]}],y [{CropImDimensions[2]}, {CropImDimensions[3]}]')

	if np.array(DarkRef).all() == None:
		DarkRef = np.zeros(shape=(CropImDimensions[0],CropImDimensions[1],3))

	## Set parameters
	DarkN = [8]  ## Indicate when to expect dark plateau when switching from panel 4 to 2
	ExpectedWavelengths = 16
	EdgeShape = EdgePos.shape
	## Check if multiple sweeps must be averaged
	if len(EdgeShape) == 3:
		Nsweep, Nplateaux, _ = EdgePos.shape
	elif len(EdgeShape) == 2:
		Nplateaux, _ = EdgePos.shape
		Nsweep = 1
		print('Only one sweep')
	else:
		print(f'There is a problem with the shape of EdgePos: {EdgeShape}')
	if Nplateaux != (ExpectedWavelengths + 1):
		print(f'Nplateaux = {Nplateaux} is not what is expected. Will run into problems.')

	## Compute Hypercube and Dark
	Hypercube = []

	bs = int(np.round(BufferSize / 2, 0))  ## Buffer size for either side of edge
	for n in range(0, Nsweep):
		## Import data
		sweep_start = EdgePos[n, 0, 0]
		sweep_length = np.sum(EdgePos[n, :, 1],axis=1)
		Video = HySE.Import.ImportData(DataPath, CropImDimensions=CropImDimensions, Coords=[sweep_start,sweep_start+sweep_length])
		Hypercube_n = []
		for i in range(0, Nplateaux):
			if i not in DarkN:  ## Skip the middle
				# if Nsweep==1:
				# 	framestart = EdgePos[i,0]
				# 	plateau_size = EdgePos[i,1]
				# else:
				framestart = EdgePos[n, i, 0]
				plateau_size = EdgePos[n, i, 1]

				s = framestart + bs
				e = framestart + plateau_size - bs
				# print(f's: {s}, e: {e}')
				s = int(s)
				e = int(e)
				## Print how many frames are averaged
				## React if number unreasonable (too small or too large)
				if i == 0:
					print(f'Computing hypercube: Averaging {e - s} frames')
				data_norm = Video[s:e, :, :]-DarkRef[None,:,:,:]
				data_norm[data_norm<=0]=0 # Remove negative values after dark subtraction
				if RefChannel:
					if Normalise:
						data_norm = np.divide(np.delete(data_norm,RefChannel,axis=3).mean(axis=3),data_norm[:,:,:,RefChannel],
											  out=np.zeros(shape=np.shape(data_norm)[0:3]),where=data_norm[:, :, :, RefChannel] != 0,
											  dtype='float32')
					else:
						data_norm = np.delete(data_norm,RefChannel,axis=3).mean(axis=3,dtype='float32')
				else:
					data_norm = data_norm.mean(axis=3,dtype='float32')
				data_avg = np.mean(data_norm, axis=0,dtype='float32')
				Hypercube_n.append(data_avg)
			else:  ## Dark
				if ComputeDarks:
					framestart = EdgePos[n, i, 0]
					plateau_size = EdgePos[n, i, 1]
					s = framestart + bs
					e = framestart + plateau_size - bs
					data_sub = Video[s:e, :, :]
					data_avg = np.mean(data_sub, axis=0)
					Darks.append(data_avg)
		Hypercube.append(Hypercube_n)
	if ComputeDarks:
		Darks = np.array(Darks)
		## Always average darks
		Darks = np.mean(Darks, axis=0)
	Hypercube = np.array(Hypercube)
	## Average sweeps
	if Average:
		Hypercube = np.mean(Hypercube, axis=0,dtype='float32')

	# Hypercube_sorted = Hypercube
	# print(f'order_list: \n{order_list}')

	## Find current path and time for saving
	time_now = datetime.now().strftime("%Y%m%d__%I-%M-%S-%p")
	day_now = datetime.now().strftime("%Y%m%d")

	## MakeFigure
	if Plot:
		if Average == False:
			HypercubeToPlot = Hypercube[0, :, :, :]
			print(f'Plotting hypercube for sweep 0')
		else:
			HypercubeToPlot = Hypercube
		nn = 0
		Mavg = np.mean(HypercubeToPlot)
		Mstd = np.std(HypercubeToPlot)
		MM = Mavg + 5 * Mstd
		fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8, 8))
		for j in range(0, 4):
			for i in range(0, 4):
				if nn < 17:
					ax[j, i].imshow(HypercubeToPlot[nn, :, :], cmap='gray')
					ax[j, i].set_title(f'im {nn}')
					ax[j, i].set_xticks([])
					ax[j, i].set_yticks([])
					nn = nn + 1
				else:
					ax[j, i].set_xticks([])
					ax[j, i].set_yticks([])
		plt.tight_layout()
		if PathToSave:
			plt.savefig(f'{PathToSave}_Hypercube.png')

	if SaveArray:
		if PathToSave:
			np.savez(f'{PathToSave}_Hypercube.npz', Hypercube)
			if ComputeDarks:
				np.savez(f'{PathToSave}_AutoDark.npz', Darks)
		else:
			print("Can't save data without PathToSave")
	if ComputeDarks:
		return Hypercube, Darks
	else:
		return Hypercube