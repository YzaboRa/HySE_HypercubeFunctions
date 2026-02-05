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
	Help = kwargs.get('Help', False)
	if Help:
		# print(info)
		print(inspect.getdoc(ComputeHypercube))
		return 0, 0

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
					RGB = HySE.UserTools.wavelength_to_rgb(wav)
					if ForCoRegistration:
						ax[j,i].imshow(HypercubeToPlot[nn,0,:,:], cmap='gray')
					else:
						ax[j,i].imshow(HypercubeToPlot[nn,:,:], cmap='gray')
					if Order:
						wav = Wavelengths_sorted[nn]
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
	Memory-efficient version of ComputeHypercube_RGB that processes frames iteratively.
	
	This version reads frames on-demand instead of loading the entire video into memory.
	Suitable for very large video files.
	
	---

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
				- Nsweep
				- SaveFig = False
				- SaveArray = False
				- Plot = True
				- NsweepOnly
				- Order = False. Set to False if doing wavelength unmixing
				- Average = False. If more than one sweep is indicated, indicates whether
					to average all sweeps before computing hypercube.
					If false, it will output as many hypercubes as sweeps.
				- ForCoRegistration = True. If True, keeps individual frames
				- BlueShift = 1. Indicates by how many frames the blue trace is ahead (since we are triggering on the red)


	Output:
	- Hypercube_sorted: Hypercube contained in a 3D array, with wavelengths sorted according
						to order_list (if Order=True)
						Shape (Nwavelengths, 1080, 1920) for HD format

	- Dark: Dark average contained in 3D array (BGR)
	"""
	import cv2

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(ComputeHypercube_RGB))
		return 0, 0
	

	Wavelengths_list = kwargs.get('Wavelengths_list', None)
	if Wavelengths_list is not None:
		## Sort Wavelengths
		order_list = np.argsort(Wavelengths_list)
		Wavelengths_sorted = Wavelengths_list[order_list]
	else:
		print(f'Provide a list of the wavelengths for ordering')
		Order=False # Can't order by wavelengths if there's no wavelength list!

	if Order:
		print(f'Order set to True. Hypercube will be sorted according to the wavelengths list')

	Name = kwargs.get('Name', '')
	SaveFig = kwargs.get('SaveFig', False)
	Order = kwargs.get('Order', False)
	SaveArray = kwargs.get('SaveArray', False)
	Plot = kwargs.get('Plot', False)
	Average = kwargs.get('Average', False)
	NsweepOnly = kwargs.get('NsweepOnly')

	
	Buffer = kwargs.get('Buffer', 6)
	print(f'Buffer of frames to ignore between neighbouring wavelengths set to 2x{Buffer}')
	BufferSize = 2 * Buffer
	
	BlueShift = kwargs.get('BlueShift', 1)
	ForCoRegistration = kwargs.get('ForCoRegistration', True)
	
	CropImDimensions = kwargs.get('CropImDimensions')
	if not CropImDimensions:
		CropImDimensions = [663, 1818, 9, 1013]
		print(f'Automatic cropping: x[{CropImDimensions[0]}:{CropImDimensions[1]}], y[{CropImDimensions[2]}:{CropImDimensions[3]}]')
	
	# Determine number of sweeps
	EdgeShape = EdgePos.shape
	if len(EdgeShape) == 3:
		Nsweep, Nplateaux, _ = EdgePos.shape
	elif len(EdgeShape) == 2:
		Nplateaux, _ = EdgePos.shape
		Nsweep = 1
	else:
		print(f'Problem with EdgePos shape: {EdgeShape}')
		return None, None
	
	DarkN = [8]
	bs = int(np.round(BufferSize / 2, 0))
	
	# Open video capture
	cap = cv2.VideoCapture(DataPath)
	if not cap.isOpened():
		raise ValueError(f"Could not open video: {DataPath}")
	
	xmin, xmax, ymin, ymax = CropImDimensions
	
	Hypercube = []
	Darks = []
	
	if NsweepOnly is None:
		nstart=0
		nend = Nsweep
	else:
		nstart = NsweepOnly
		nend = NsweepOnly+1
	

	for n in range(nstart, nend):
		Hypercube_n = []
		
		for i in range(Nplateaux):
			framestart = EdgePos[n, i, 0]
			plateau_size = EdgePos[n, i, 1]
			
			s = int(framestart + bs)
			e = int(framestart + plateau_size - bs)
			
			if i == 0:
				print(f'Sweep {n}, processing {e - s} frames per wavelength')
			
			# Read frames for this plateau
			frames_list_r = []
			frames_list_g = []
			frames_list_b = []
			
			for frame_idx in range(s, e):
				cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
				ret, frame = cap.read()
				
				if not ret:
					print(f"Warning: Could not read frame {frame_idx}")
					continue
				
				# Crop frame
				frame_cropped = frame[ymin:ymax, xmin:xmax]
				
				# Split BGR channels
				frames_list_b.append(frame_cropped[:, :, 0])
				frames_list_g.append(frame_cropped[:, :, 1])
				frames_list_r.append(frame_cropped[:, :, 2])
			
			# Handle BlueShift for non-dark plateaus
			if i not in DarkN and BlueShift > 0:
				# Read additional blue frames
				for frame_idx in range(e, e + BlueShift):
					cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
					ret, frame = cap.read()
					if ret:
						frame_cropped = frame[ymin:ymax, xmin:xmax]
						frames_list_b.append(frame_cropped[:, :, 0])
				
				# Adjust blue list to match shift
				frames_list_b = frames_list_b[BlueShift:]
			
			# Convert to arrays
			data_sub_r = np.array(frames_list_r)
			data_sub_g = np.array(frames_list_g)
			data_sub_b = np.array(frames_list_b)
			
			# Average or keep individual frames
			if ForCoRegistration:
				data_avg = np.stack((data_sub_b, data_sub_g, data_sub_r), axis=-1)
			else:
				avg_b = np.average(data_sub_b, axis=0)
				avg_g = np.average(data_sub_g, axis=0)
				avg_r = np.average(data_sub_r, axis=0)
				data_avg = np.stack((avg_b, avg_g, avg_r), axis=-1)
			
			if i not in DarkN:
				Hypercube_n.append(data_avg)
			else:
				if not ForCoRegistration:
					# For dark frames, average all channels
					dark_avg = np.mean(data_avg, axis=-1, keepdims=True)
					dark_avg = np.repeat(dark_avg, 3, axis=-1)
					Darks.append(dark_avg)
		
		Hypercube.append(Hypercube_n)
	
	cap.release()
	
	Hypercube = np.array(Hypercube)
	Darks = np.array(Darks) if Darks else None
	
	if Darks is not None:
		Darks = np.average(Darks, axis=0)


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
			print(Hypercube.shape)
			Nsweeps, WW, Nframes, YY, XX, _ = Hypercube.shape
			## sort Hypercube
			# Hypercube_sorted = np.array(Hypercube_sorted)

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
	
	return Hypercube, Darks



def ComputeHypercube_RGB_orig(DataPath, EdgePos, **kwargs):
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
				- Nsweep
				- SaveFig = False
				- SaveArray = False
				- Plot = True
				- Order = False. Set to False if doing wavelength unmixing
				- Average = False. If more than one sweep is indicated, indicates whether
					to average all sweeps before computing hypercube.
					If false, it will output as many hypercubes as sweeps.
				- ForCoRegistration = True. If True, keeps individual frames
				- BlueShift = 1. Indicates by how many frames the blue trace is ahead (since we are triggering on the red)


	Output:
	- Hypercube_sorted: Hypercube contained in a 3D array, with wavelengths sorted according
						to order_list (if Order=True)
						Shape (Nwavelengths, 1080, 1920) for HD format

	- Dark: Dark average contained in 3D array (BGR)


	"""

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(ComputeHypercube_RGB_orig))
		return 0, 0


	## Check if the user has set the Buffer size

	Buffer = kwargs.get('Buffer', 6)
	print(f'Buffer of frames to ignore between neighbouring wavelenghts set to 2x{Buffer}')
	BufferSize = 2*Buffer
	
	BlueShift = kwargs.get('BlueShift', 1)
	Nsweep = kwargs.get('Nsweep')

	Name = kwargs.get('Name', '')
	SaveFig = kwargs.get('SaveFig', False)
	Order = kwargs.get('Order', False)

	Wavelengths_list = kwargs.get('Wavelengths_list', None)
	if Wavelengths_list is not None:
		## Sort Wavelengths
		order_list = np.argsort(Wavelengths_list)
		Wavelengths_sorted = Wavelengths_list[order_list]
	else:
		print(f'Provide a list of the wavelengths for ordering')
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
	# data = HySE.ImportData(DataPath, CropImDimensions=CropImDimensions, RGB=True)

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
				if BlueShift==0:
					data_sub = data[s:e,:,:,:]
					if ForCoRegistration:
						data_avg = data_sub
					else:
						data_avg = np.average(data_sub, axis=0)
				else:
					if i==0:
						print(f'Shifting blue channel by {BlueShift}')
					data_sub_red = data[s:e,:,:,2]
					data_sub_green = data[s:e,:,:,1]
					data_sub_blue = data[(s+BlueShift):(e+BlueShift),:,:,0]
					if ForCoRegistration:
						## Combine data_sub_red, data_sub_green, data_sub_blue in a single array
						## of size [Nframes, YY, XX, RGB_3]
						data_avg = np.stack((data_sub_blue, data_sub_green, data_sub_red), axis=-1)
					else:
						# First, average each color channel across the time (frame) axis.
						avg_blue = np.average(data_sub_blue, axis=0)
						avg_green = np.average(data_sub_green, axis=0)
						avg_red = np.average(data_sub_red, axis=0)

						# Then, stack the resulting 2D arrays into a single 3D color image.
						data_avg = np.stack((avg_blue, avg_green, avg_red), axis=-1)
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
			print(Hypercube.shape)
			Nsweeps, WW, Nframes, YY, XX, _ = Hypercube.shape
			## sort Hypercube
			# Hypercube_sorted = np.array(Hypercube_sorted)

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


# def ComputeHypercube_RGB(DataPath, EdgePos, **kwargs):
# 	"""
# 	Function to compute the hypercube. It inputs the path to the data and
# 	the EdgePos output from the FindHypercube function (which indicates where
# 	to find the start for each wavelenght for each identified sweep)
	
# 	Input:
# 	- DataPath: Path to the data
	
# 	- EdgePos: Position of the start of each colour for each sweep, output of FindHypercube
# 			   Functions are separated to allow tweaking of parameters to properly identify 
# 			   individual sweeps
	
# 	- kwargs (optional): Optional parameters
# 				- Help = False 
# 				- Wavelengths_list: List of wavelengths as measured in the data
# 				- Buffer = integer : Number of frames to ignore between neighbouring 
# 					colours to avoid contamination by transition frames.
# 					Might need to be adjusted for very short or very large repetitions.
# 					Default to 6 
# 				- Name = string
# 				- SaveFig = False
# 				- SaveArray = False
# 				- Plot = True
# 				- Order = False. Set to False if doing wavelength unmixing
# 				- Average = False. If more than one sweep is indicated, indicates whether
# 					to average all sweeps before computing hypercube.
# 					If false, it will output as many hypercubes as sweeps.
# 				- ForCoRegistration = True. If True, keeps individual frames
										 
	
# 	Output:
# 	- Hypercube_sorted: Hypercube contained in a 3D array, with wavelengths sorted according
# 						to order_list (if Order=True)
# 						Shape (Nwavelengths, 1080, 1920) for HD format
						
# 	- Dark: Dark average contained in 3D array (BGR)
	
	
# 	"""

# 	Help = kwargs.get('Help', False)
# 	if Help:
# 		print(inspect.getdoc(ComputeHypercube_RGB))
# 		return 0, 0


# 	## Check if the user has set the Buffer size

# 	Buffer = kwargs.get('Buffer', 6)
# 	print(f'Buffer of frames to ignore between neighbouring wavelenghts set to 2x{Buffer}')
# 	BufferSize = 2*Buffer
	
# 	Name = kwargs.get('Name', '')
# 	SaveFig = kwargs.get('SaveFig', False)
# 	Order = kwargs.get('Order', False)

# 	Wavelengths_list = kwargs.get('Wavelengths_list', None)
# 	if Wavelengths_list:
# 	   ## Sort Wavelengths
# 	   order_list = np.argsort(Wavelengths_list)
# 	   Wavelengths_sorted = Wavelengths_list[order_list]
# 	   # print(f'Wavelengths_list: \n{Wavelengths_list}\n\n')
# 	   # print(f'Wavelengths_sorted: \n{Wavelengths_sorted}\n\n')
# 	else:
# 	   Order=False # Can't order by wavelengths if there's no wavelength list!

# 	if Order:
# 		print(f'Order set to True. Hypercube will be sorted according to the wavelengths list')


# 	SaveArray = kwargs.get('SaveArray', False)
	
# 	Plot = kwargs.get('Plot', False)
# 	Average = kwargs.get('Average', False)

# 	ForCoRegistration = kwargs.get('ForCoRegistration', True)
# 	if ForCoRegistration:
# 		print(f'Keeping individual frames for co-registration')

# 	CropImDimensions = kwargs.get('CropImDimensions')
# 	if not CropImDimensions:
# 		# CropImDimensions = [702,1856, 39,1039]
# 		# ## [702,1856, 39,1039] ## xmin, xmax, ymin, ymax - CCRC SDI full canvas
# 		# ## [263,695, 99,475] ## xmin, xmax, ymin, ymax  - CCRC standard canvas
# 		CropImDimensions = [663,1818,9,1013]
# 		## [702,1856, 39,1039] ## xmin, xmax, ymin, ymax - CCRC SDI full canvas
# 		## [263,695, 99,475] ## xmin, xmax, ymin, ymax  - CCRC standard canvas
# 		## [663,1818,9,1013] ## xmin, xmax, ymin, ymax  - CCRC standard canvas since August 2025
# 		print(f'Automatic cropping: [{CropImDimensions[0]} : {CropImDimensions[1]}],y [{CropImDimensions[2]}, {CropImDimensions[3]}]')
# 	else:
# 		print(f'Cropping image: x [{CropImDimensions[0]} : {CropImDimensions[1]}],y [{CropImDimensions[2]}, {CropImDimensions[3]}]')

		
# 	## Import data
# 	data = HySE.Import.ImportData(DataPath, CropImDimensions=CropImDimensions, RGB=True)

	
# 	## Set parameters
# 	DarkN = [8] ## Indicate when to expect dark plateau when switching from panel 4 to 2
# 	ExpectedWavelengths = 16
# 	EdgeShape = EdgePos.shape
# 	## Check if multiple sweeps must be averaged
# 	if len(EdgeShape)==3:
# 		Nsweep, Nplateaux, _ = EdgePos.shape
# 	elif len(EdgeShape)==2:
# 		Nplateaux, _ = EdgePos.shape
# 		Nsweep = 1
# 		print('Only one sweep')
# 	else:
# 		print(f'There is a problem with the shape of EdgePos: {EdgeShape}')
# 	if Nplateaux!=(ExpectedWavelengths+1):
# 		print(f'Nplateaux = {Nplateaux} is not what is expected. Will run into problems.')
	
# 	## Compute Hypercube and Dark
# 	Hypercube = []
# 	Darks = []
# 	bs = int(np.round(BufferSize/2,0)) ## Buffer size for either side of edge
# 	for n in range(0,Nsweep):
# 		Hypercube_n = []
# 		for i in range(0,Nplateaux):
# 			if i not in DarkN: ## Skip the middle
# 				# if Nsweep==1:
# 				# 	framestart = EdgePos[i,0]
# 				# 	plateau_size = EdgePos[i,1]
# 				# else:
# 				framestart = EdgePos[n,i,0]
# 				plateau_size = EdgePos[n,i,1]

# 				s = framestart+bs
# 				e = framestart+plateau_size-bs
# 				# print(f's: {s}, e: {e}')
# 				s = int(s)
# 				e = int(e)
# 				## Print how many frames are averaged
# 				## React if number unreasonable (too small or too large)
# 				if i==0: 
# 					if ForCoRegistration==False:
# 						print(f'Computing hypercube: Averaging {e-s} frames')
# 				data_sub = data[s:e,:,:,:]
# 				if ForCoRegistration:
# 					data_avg = data_sub
# 				else:
# 					data_avg = np.average(data_sub, axis=0)
# 				Hypercube_n.append(data_avg)
# 			else: ## Dark
# 				# if Nsweep==1:
# 				# 	framestart = EdgePos[i,0]
# 				# 	plateau_size = EdgePos[i,1]
# 				# else:
# 				framestart = EdgePos[n,i,0]
# 				plateau_size = EdgePos[n,i,1]

# 				s = framestart+bs
# 				e = framestart+plateau_size-bs
# 				data_sub = data[s:e,:,:,:]
# 				data_avg = np.average(data_sub, axis=0)
# 				Darks.append(data_avg) 
# 		Hypercube.append(Hypercube_n)
# 	Darks = np.array(Darks)
# 	Hypercube = np.array(Hypercube)
# 	## Average sweeps
# 	if Average:
# 		Hypercube = np.average(Hypercube,axis=0)
# 	## Always average darks
# 	Darks = np.average(Darks,axis=0)
	
# 	# Sort hypercube according to the order_list
# 	# Ensures wavelenghts are ordered from blue to red
# 	if Order:
# 		if ForCoRegistration:
# 			print(f'Ordering hypercube when keeping all frames (ForCoRegistration = True) is not supported.')
# 		if Average:
# 			Hypercube_sorted = []
# 			for k in range(0,Hypercube.shape[0]):
# 				Hypercube_sorted.append(Hypercube[order_list[k]])
# 			Hypercube_sorted = np.array(Hypercube_sorted)
# 		else:
# 			print(f'Warning: in HySE.ManipulateHypercube.ComputeHypercube(), Order=True but Average=False.')
# 			Hypercube_sorted = []
# 			NN, WW, YY, XX = Hypercube.shape
# 			for n in range(0,NN):
# 				hypercube_sorted_sub = []
# 				for k in range(0,WW):
# 					hypercube_sorted_sub.append(Hypercube[n,order_list[k],:,:,:])
# 				Hypercube_sorted.append(hypercube_sorted_sub)
# 			Hypercube_sorted = np.array(Hypercube_sorted)

# 	else:
# 		Hypercube_sorted = Hypercube

# 	# Hypercube_sorted = Hypercube
# 	# print(f'order_list: \n{order_list}')

# 	## Find current path and time for saving
# 	time_now = datetime.now().strftime("%Y%m%d__%I-%M-%S-%p")
# 	day_now = datetime.now().strftime("%Y%m%d")

# 	Name_withExtension = DataPath.split('/')[-1]
# 	Name = Name_withExtension.split('.')[0]
# 	Path = DataPath.replace(Name_withExtension, '')
	
# 	## MakeFigure
# 	if Plot:
# 		if Average==False:
# 			HypercubeToPlot = Hypercube_sorted[0,:,:,:,1]
# 			print(f'Plotting hypercube for sweep 0, green frame')
# 		else: 
# 			HypercubeToPlot = Hypercube_sorted[:,:,1]
# 			print(f'Plotting hypercube for green frame')
# 		nn = 0
# 		Mavg = np.average(HypercubeToPlot)
# 		Mstd = np.std(HypercubeToPlot)
# 		MM = Mavg+5*Mstd
# 		fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
# 		for j in range(0,4):
# 			for i in range(0,4):
# 				if nn<17:
# 					wav = Wavelengths_sorted[nn]
# 					RGB = HySE.UserTools.wavelength_to_rgb(wav)
# 					if ForCoRegistration:
# 						ax[j,i].imshow(HypercubeToPlot[nn,0,:,:], cmap='gray')
# 					else:
# 						ax[j,i].imshow(HypercubeToPlot[nn,:,:], cmap='gray')
# 					if Order:
# 						ax[j,i].set_title(f'{wav} nm', c=RGB)
# 					else:
# 						ax[j,i].set_title(f'im {nn}')
# 					ax[j,i].set_xticks([])
# 					ax[j,i].set_yticks([])
# 					nn = nn+1
# 				else:
# 					ax[j,i].set_xticks([])
# 					ax[j,i].set_yticks([])
# 		plt.tight_layout()
# 		if SaveFig:
# 			PathToSave = f'{Path}{time_now}_{Name}'
# 			plt.savefig(f'{PathToSave}_Hypercube.png')
# 			print(f'Saved figure at {PathToSave}')

# 	if SaveArray:
# 		PathToSave = f'{Path}{time_now}_{Name}'
# 		np.savez(f'{PathToSave}_Hypercube.npz', Hypercube_sorted)
# 		np.savez(f'{PathToSave}_AutoDark.npz', Darks)
# 	return Hypercube_sorted, Darks





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
	# dark = np.average(frames,axis=0)
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

