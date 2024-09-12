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




ef NormaliseHypercube(DataPath, Hypercube, Hypercube_White, Dark, Wavelengths_list, **kwargs):
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




def GetDark(DataAll, EdgePos, **kwargs):
	try:
		Buffer = kwargs['Buffer']
		print(f'Buffer set to {Buffer}.')
	except KeyError:
		Buffer = 6
		print(f'Buffer not specified. Set to default 6.')

	try:
		DarkRepeat = kwargs['DarkRepeat']
		print(f'DarkRepeat set to {DarkRepeat}.')
	except KeyError:
		DarkRepeat = 3
		print(f'Assuming DarkRepeat = 3,')

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
	return DarkAvg
