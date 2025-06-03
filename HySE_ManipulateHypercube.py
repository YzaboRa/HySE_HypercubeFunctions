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
# import matplotlib.colors as colors
# import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"


import HySE_ImportData
import HySE_UserTools




def ComputeHypercube(DataPath, EdgePos, Wavelengths_list, **kwargs):
	info="""
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
				- Help = False 
				- Buffer = integer : Number of frames to ignore between neighbouring 
					colours to avoid contamination by transition frames.
					Might need to be adjusted for very short or very large repetitions.
					Default to 6 
				- Name = string
				- SaveFig = True
				- SaveArray = True
				- PlotHypercube = True
				- Order = True. Set to False if doing wavelength unmixing
										 
	
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
	Order = kwargs.get('Order', True)
	if Order==False:
		print(f'Order set to False: the hypercube output will be out of order. Use for spectral unmixing.')
	SaveArray = kwargs.get('SaveArray', True)
	Help = kwargs.get('Help', False)
	if Help:
		print(info)
		return 0, 0
	PlotHypercube = kwargs.get('PlotHypercube', False)

	CropImDimensions = kwargs.get('CropImDimensions')
	if not CropImDimensions:
		CropImDimensions = [702,1856, 39,1039]
		## [702,1856, 39,1039] ## xmin, xmax, ymin, ymax - CCRC SDI full canvas
		## [263,695, 99,475] ## xmin, xmax, ymin, ymax  - CCRC standard canvas
		print(f'Automatic cropping: [{CropImDimensions[0]} : {CropImDimensions[1]}],y [{CropImDimensions[2]}, {CropImDimensions[3]}]')
	else:
		print(f'Cropping image: x [{CropImDimensions[0]} : {CropImDimensions[1]}],y [{CropImDimensions[2]}, {CropImDimensions[3]}]')

		
	## Import data
	data = HySE_ImportData.ImportData(DataPath, CropImDimensions=CropImDimensions)
	
	## Sort Wavelengths
	order_list = np.argsort(Wavelengths_list)
	Wavelengths_sorted = Wavelengths_list[order_list]
	# print(f'Wavelengths_list: \n{Wavelengths_list}\n\n')
	# print(f'Wavelengths_sorted: \n{Wavelengths_sorted}\n\n')

	
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
					print(f'Computing hypercube: Averaging {e-s} frames')
				data_sub = data[s:e,:,:]
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
	Hypercube = np.average(Hypercube,axis=0)
	Darks = np.average(Darks,axis=0)
	
	# Sort hypercube according to the order_list
	# Ensures wavelenghts are ordered from blue to red
	if Order:
		Hypercube_sorted = []
		for k in range(0,Hypercube.shape[0]):
			Hypercube_sorted.append(Hypercube[order_list[k]])
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
	if PlotHypercube:
		nn = 0
		Mavg = np.average(Hypercube_sorted)
		Mstd = np.std(Hypercube_sorted)
		MM = Mavg+5*Mstd
		fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
		for j in range(0,4):
			for i in range(0,4):
				if nn<17:
					wav = Wavelengths_sorted[nn]
					RGB = HySE_UserTools.wavelength_to_rgb(wav)
					ax[j,i].imshow(Hypercube_sorted[nn,:,:], cmap='gray')
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

	if SaveArray:
		np.savez(f'{PathToSave}_Hypercube.npz', Hypercube_sorted)
		np.savez(f'{PathToSave}_AutoDark.npz', Darks)
	return Hypercube_sorted, Darks






def NormaliseHypercube_Old(DataPath, Hypercube, Hypercube_White, Dark, Wavelengths_list, **kwargs):
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
				RGB = HySE_UserTools.wavelength_to_rgb(wav)
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



def NormaliseFrames(image, image_white, image_dark):
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



def GetDark(vidPath, EdgePos, **kwargs):
	'''
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

	'''
	DataAll = HySE_ImportData.ImportData(vidPath, **kwargs)
	DarkAvg = GetDark_FromData(DataAll, EdgePos, **kwargs)
	return DarkAvg


def GetDark_FromData(DataAll, EdgePos, **kwargs):

	Buffer = kwargs.get('Buffer', 6)

	try:
		DarkRepeat = kwargs['DarkRepeat']
		print(f'DarkRepeat set to {DarkRepeat}.')
	except KeyError:
		DarkRepeat = 3
		print(f'Assuming DarkRepeat = 3,')

	SaveDark = kwargs.get('SaveDark', True)

	try: 
		SavePath = kwargs['SavePath']
		print(f'SavePath set to {SavePath}.')
	except KeyError:
		SavePath = ''
		if SaveDark:
			print(f'Please input a saving path with SavePath=\'\'')

	## Automatic dark checks parameters
	std_Max = 5
	avg_Max = 25


	## Number of sweeps
	NNsweeps = len(EdgePos)

	## Size of the dataset
	(NN, YY, XX) = DataAll.shape

	AllDarks = []
	## Loop through all of the sweeps
	for n in range(0,NNsweeps):
		start = EdgePos[n][-1][0] + EdgePos[n][-1][1] + Buffer

		if n==(NNsweeps-1):
			# print(f'	n={n}, (NN-2)={NNsweeps-2}')
			## If this is the last sweep, check if we need to go to end of the sweep
			## or estimate the size of a long dark (to avoid start of unfinished sweep)
			end_estimate = start+ np.amin(EdgePos[n][:,1])*DarkRepeat
			end_sweep = NN
			# print(f'	end_estimate={end_estimate}, end_sweep={end_sweep}')
			end = min(end_estimate, end_sweep)
		else:
			# print(f'	n={n}')
			end = EdgePos[n+1][0][0] - Buffer

		# print(f'	n={n}, start={start}, end={end}')

		## Select frames from the long dark
		frames = DataAll[start:end, :,:]
		m, M = np.amin(frames), np.amax(frames)
		avg, std = np.average(frames), np.std(frames)
		print(f'min = {m:.2f}, max = {M:.2f}, avg = {avg:.2f}, std = {std:.2f}')

		## Add extra step to make sure we are not including non-dark frames
		if (std>std_Max or avg>avg_Max):
			print(f'It seems like there are outlier parameters in the dark frames')
			print(f'	min = {m:.2f}, max = {M:.2f}, avg = {avg:.2f}, std = {std:.2f}')
			print(f'	Use \'DarkRepeat\' and \'Buffer\' to adjust the dark selection')


		dark = np.average(frames,axis=0)
		# print(f'dark.shape = {dark.shape}')
		AllDarks.append(dark)

	AllDarks = np.array(AllDarks)
	DarkAvg = np.average(AllDarks,axis=0)

	if SaveDark:
		if '.npz' not in SavePath:
			SavePath_Dark = SavePath+'_Dark.npz'
		else:
			SavePath_Dark.replace('.npz','_Dark.npz')
		np.savez(SavePath_Dark, DarkAvg)
	return DarkAvg
