"""

Functions that represent tools for the user (plotting, saving, help, etc.)

"""


import numpy as np
import cv2
import os
from datetime import datetime
from scipy.signal import savgol_filter, find_peaks
import matplotlib
from matplotlib import pyplot as plt
import imageio
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math
import inspect
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"





def FindPlottingRange(array):
	"""
	Function that helps finding a reasonable range for plotting 
	Helpful when the data has several pixels with abnormally high/low values such that 
	the automatic range does not allow to visualise data (frequent after normalisation, when
	some areas of the image are dark)

	Input:
		- array (to plot)
		- kwargs:
			- std_range (default 3)
			- std_max_range 
			- std_min_range
			- Help

	Output:
		- m: Min value for plotting (vmin=m)
		- M: Max value for plitting (vmax=M)

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(FindPlottingRange))
		return 0,0

	std_range = kwargs.get('std_range', 3)
	std_max_range = kwargs.get('std_max_range', std_range)
	std_min_range = kwargs.get('std_min_range', std_range)

	array_flat = array.flatten()
	array_sorted = np.sort(array_flat)    
	mean = np.average(array_sorted)
	std = np.std(array_sorted)
	MM = mean + std_max_range*std
	mm = mean - std_min_range*std
	return mm, MM


def find_closest(arr, val):
	"""
	Function that finds the index in a given array whose value is the closest to a provided value

	Input:
		- arr: Array from which the index will be pulled from
		- val: Value to match as closely as possible

	Outout:
		- idx: Index from the provided array whose value is the closest to provided value

	"""
	idx = np.abs(arr - val).argmin()
	return idx



def wavelength_to_rgb(wavelength, gamma=0.8, **kwargs):

	'''This converts a given wavelength of light to an 
	approximate RGB color value. The wavelength must be given
	in nanometers in the range from 380 nm through 750 nm
	(789 THz through 400 THz).
	Based on code by Dan Bruton
	http://www.physics.sfasu.edu/astro/color/spectra.html

	Input:
		- wavelength (in nm)
		- gamma (default 0.8): transparancy value

	Return:
		- (R, G, B) value corresponding to the colour of the wavelength
	'''

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(wavelength_to_rgb))
		return (0,0,0)

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
	Function that produces a figure showing the co-registration of a given shifted image.

	Input:
		- im_static
		- im_shifted
		- im_coregistered
		- kwargs:
			- Help
			- ShowPlot (default True)
			- SavePlot (default False)
			- SavingPathWithName (default ''): If Saving figure, indicate the path where to save it
				Include the full name and '.png'.

	Output:
		- (Plots figure)

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(PlotCoRegistered))

	kwargs.get('SavingPathWithName', '')
	kwargs.get('SavePlot', False)
	kwargs.get('ShowPlot', True)


	images_diff_0 = np.subtract(im_shifted.astype('float64'), im_static.astype('float64'))
	images_diff_0_avg = np.average(np.abs(images_diff_0))
#     images_diff_0_std = np.std(np.abs(images_diff_0))
	images_diff_cr = np.subtract(im_coregistered.astype('float64'), im_static.astype('float64'))
	images_diff_cr_avg = np.average(np.abs(images_diff_cr))
#     images_diff_cr_std = np.average(np.std(images_diff_cr))

#     mmm, MMM = 0, 255
	mmm = min(np.amin(im_static), np.amin(im_shifted), np.amin(im_coregistered))
	MMM = max(np.amax(im_static), np.amax(im_shifted), np.amax(im_coregistered))
	mm0, MM0 = FindPlottingRange(images_diff_0)
	mm, MM = FindPlottingRange(images_diff_cr)

	norm = MidpointNormalize(vmin=mm0, vmax=MM0, midpoint=0)
	cmap = 'RdBu_r'

	m, M = FindPlottingRange(im_static)
	fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12,7))
	im00 = ax[0,0].imshow(im_static, cmap='gray',vmin=m, vmax=M)
	ax[0,0].set_title('Static Image')
	divider = make_axes_locatable(ax[0,0])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im00, cax=cax, orientation='vertical')

	m, M = FindPlottingRange(im_shifted)
	im01 = ax[0,1].imshow(im_shifted, cmap='gray',vmin=m, vmax=M)
	ax[0,1].set_title('Shifted Image')
	divider = make_axes_locatable(ax[0,1])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im01, cax=cax, orientation='vertical')

	im02 = ax[0,2].imshow(images_diff_0, cmap=cmap, norm=norm)
	ax[0,2].set_title(f'Difference (no registration)\n avg {images_diff_0_avg:.2f}')
	divider = make_axes_locatable(ax[0,2])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im02, cax=cax, orientation='vertical')

	m, M = FindPlottingRange(im_static)
	im10 = ax[1,0].imshow(im_static, cmap='gray',vmin=m, vmax=M)
	ax[1,0].set_title('Static Image')
	divider = make_axes_locatable(ax[1,0])
	cax = divider.append_axes('right', size='5%', pad=0.05)
	cbar = fig.colorbar(im10, cax=cax, orientation='vertical')

	m, M = FindPlottingRange(im_coregistered)
	im11 = ax[1,1].imshow(im_coregistered, cmap='gray',vmin=m, vmax=M)
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
			- Masks
			- SavePlot: (default False)
			- SavingPathWithName: Where to save the plot if SavePlot=True
			- ShowPlot: (default True)
			- SameScale (default False)
			- vmax
			- Help

	Output:
		- Figure (4x4, one wavelength per subfigure)
		Saved:
		if SavePlot=True:
			Figure

	"""

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(PlotHypercube))
		return 0

	Wavelengths = kwargs.get('Wavelengths')
	if Wavelengths is None:
		Wavelengths = [0]
		print("Input 'Wavelengths' list for better plot")

	SavePlot = kwargs.get('SavePlot', False)
	SavingPathWithName = kwargs.get('SavingPathWithName')
	if SavingPathWithName is None:
		SavingPathWithName = ''
		if SavePlot:
			print(f'SavePlot is set to True. Please input a SavingPathWithName')

	ShowPlot = kwargs.get('ShowPlot', True)
	SameScale = kwargs.get('SameScale', False)
	vmax = kwargs.get('vmax')
	if vmax is not None:
		SameScale = True
	Masks = kwargs.get('Masks')
	if Masks is None:
		MaskPlots = False
	else:
		MaskPlots = True

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

				if MaskPlots:
					array = Hypercube[nn,:,:]
					if len(Masks.shape)==2:
						mask = Masks
					elif len(Masks.shape)==3:
						mask = Masks[nn,:,:]
					else:
						print(f'Masks shape error:  {Masks.shape}')
						return 0
					ArrayToPlot = np.ma.array(array, mask=mask)
				else:
					ArrayToPlot = Hypercube[nn,:,:]

				if SameScale:
					if vmax is None:
						ax[j,i].imshow(ArrayToPlot, cmap='gray', vmin=0, vmax=np.amax(Hypercube))
					else:
						ax[j,i].imshow(ArrayToPlot, cmap='gray', vmin=0, vmax=vmax)
				else:
					ax[j,i].imshow(ArrayToPlot, cmap='gray', vmin=0, vmax=np.average(ArrayToPlot)*3)
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

	plt.tight_layout()
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
			- Help
	Output:
		Saved:
			mp4 video
	'''
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(MakeHypercubeVideo))
	
	fps = kwargs.get('fps', 10)

	(NN, YY, XX) = Hypercube.shape
	if '.mp4' not in SavingPathWithName:
		SavingPathWithName = SavingPathWithName+'.mp4'

	out = cv2.VideoWriter(SavingPathWithName, cv2.VideoWriter_fourcc(*'mp4v'), fps, (XX, YY), False)
	for i in range(NN):
		data = Hypercube[i,:,:].astype('uint8')
		out.write(data)
	out.release()


def PlotDark(Dark):
	"""
	Function that plots the dark reference for inspection

	Inputs:
		- Dark

	Outputs:
		- (plot figure)

	"""
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
	im = ax.imshow(Dark, cmap='gray')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)
	ax.set_title(f'Dark \navg: {np.average(Dark):.2f}, min: {np.nanmin(Dark):.2f}, max: {np.nanmax(Dark):.2f}')
	plt.show()




### Macbeth colour charts

def GetPatchPos(Patch1_pos, Patch_size_x, Patch_size_y, Image_angle, **kwargs):
	"""
	Function that estimates the position of each patch in the macbeth chart. 
	It identifies a corner in the patch, and defines a square region that should sit within the regions of the patch.
	If the squares do not fit nicely in the patches, play with the different parameters.
	To be used in testing/calibratin datasets done by imaging a standard macbeth colourchart.
	The output is designed to be checked with PlotPatchesDetection() and used with GetPatchesSpectrum() functions.

	Inputs:
		- Patch1_pos [y0,x0]: Coordinates of patch 1 (brown, corner)
		- Patch_size_x: Estimate (in pixels) of the spacing between patches in the x axis
		- Patch_size_y: Estimate (in pixels) of the spacing between patches in the y axis
		- Image_angle: angle (in degrees) of the chart in the image
		- kwargs:
			- Help

	Outputs:
		- Positions: Array containing the coordinates for each of the 30 patches 

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetPatchPos))

	Positions = []
	[y0, x0] = Patch1_pos
	Image_angle_rad = Image_angle*np.pi/180
	index = 0
	for j in range(0,5):
		y0s = y0 -  j*Patch_size_y*np.cos(Image_angle_rad) #j*Patch_size -
		x0s = x0 + j*Patch_size_x*np.sin(Image_angle_rad)
		for i in range(0,6):
			x = x0s - Patch_size_x*np.cos(Image_angle_rad)*i
			y = y0s - Patch_size_x*np.sin(Image_angle_rad)*i
			if (j==0 and i==5):
				y = y-15
				x = x+10
			Positions.append([index, x, y])
			index +=1
	return np.array(Positions)

def GetPatchesSpectrum(Hypercube, Sample_size, Positions, CropCoordinates, **kwargs):
	"""
	Function that extracts the average spectrum in each patch region, as defined by the output from the GetPatchPos() function.

	Inputs:
		- Hypercube
		- Sample_size: size of a patch
		- Positions: Positions of each patch, output from GetPatchPos()
		- CropCoordingates: For the full image
		- kwargs:
			- Help

	Output:
		- Spectra: Array containing the average spectra for all patches

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetPatchesSpectrum))

	Spectrum = []
	(NN, YY, XX) = Hypercube.shape
	for n in range(0,NN):
		im_sub = Hypercube[n,:,:]
		Intensities = GetPatchesIntensity(im_sub[CropCoordinates[2]:CropCoordinates[3], CropCoordinates[0]:CropCoordinates[1]], Sample_size, Positions)
		Spectrum.append(Intensities)
	return np.array(Spectrum)


def GetPatchesIntensity(Image, Sample_size, PatchesPositions):
	"""
	Functino used in GetPatchesSpectrum() to calculate the average value (for a single wavelength/image) for all patches

	Inputs:
		- Image (slide of the hypercube, single wavelength)
		- Sample_size: estimated size of a patch (smaller than real patch to avoid unwanted pixels)
		- PatchesPositions: Positions of each patches, output of GetPatchPos()

	Outputs:
		- Intensities: Array size 30 containing the average intensity for this given image/wavelenght for all patches

	"""
	N = len(PatchesPositions)
	Intensities = []
	for n in range(0,N):
		nn = PatchesPositions[n,0]
		x0, y0 = PatchesPositions[n,1], PatchesPositions[n,2]
		xs, xe  = int(x0-Sample_size/2), int(x0+Sample_size/2)
		ys, ye  = int(y0-Sample_size/2), int(y0+Sample_size/2)
		im_sub = Image[ys:ye, xs:xe]
		val = np.average(im_sub)
		std = np.std(im_sub)
		Intensities.append([nn, val, std])       
	return np.array(Intensities)


def PlotPatchesDetection(macbeth, Positions, Sample_size):
	"""
	Function that plots the automatic patch position estimates over an image of the macbeth chart (from the data)
	Use this to make sure that the patches have been properly identified and that all the pixels included 
	are indeed part of the patch, to avoid corrupted spectra

	Inputs:
		- macbeth: monochromatic image of the macbeth chart (from the data, same dimensions)
		- Positions: Positions of each patches, output of GetPatchPos()
		- Sample_size: estimated size of a patch (smaller than real patch to avoid unwanted pixels)

	Outputs:
		- (plots figure)


	"""
	fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
	im = ax.imshow(macbeth, cmap='gray')
	for i in range(0, 30):  
		ax.scatter(Positions[i,1], Positions[i,2], color='cornflowerblue', s=15)
		ax.text(Positions[i,1]-10, Positions[i,2]-8, f'{Positions[i,0]+1:.0f}', color='red')
		area = patches.Rectangle((Positions[i,1]-Sample_size/2, Positions[i,2]-Sample_size/2), Sample_size, Sample_size, edgecolor='none', facecolor='cornflowerblue', alpha=0.4)
		ax.add_patch(area)
	plt.tight_layout()
	plt.show()



def psnr(img1, img2, **kwargs):
	"""

	Function that computes the peak signal to noise ratio (PSNR) between two images
	Used to calculate how closely data matches a reference (spectra)

	Inputs:
		- img1
		- img2
		- kwargs:
			- Help

	Outputs:
		- psnr

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(psnr))

	mse = np.mean(np.square(np.subtract(img1,img2)))
	if mse==0:
		return np.Inf
	max_pixel = 1 #255.0
	psnr = 20 * math.log10(max_pixel / np.sqrt(mse)) 
#     psnr = 20 * math.log10(max_pixel) - 10 * math.log10(mse)  
	return psnr


def CompareSpectra(Wavelengths_sorted, GroundTruthWavelengths, GroundTruthSpectrum, **kwargs):
	"""
	Function that outputs a grout truth reference spectra of the same size as the data 
	Allows to plot both on the same x data list. 
	Assumes that the ground truth has more wavelengths than the dataset

	Inputs:
		- Wavelengths_sorted: list of wavelengths (data)
		- GroundTruthWavelengths: list of wavelengths (ground truth/reference)
		- GroundTruthSpectrum: Spectra of the ground truth/reference (same length as GroundTruthWavelengths)
		- kwargs:
			- Help

	Output:
		- Comparable_GroundTruthSpectrum


	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(CompareSpectra))

	ComparableSpectra = []
	for i in range(0,len(Wavelengths_sorted)):
		wav=Wavelengths_sorted[i]
		index=find_closest(GroundTruthWavelengths, wav)
		ComparableSpectra.append(GroundTruthSpectrum[index])
	return np.array(ComparableSpectra)


def PlotPatchesSpectra(PatchesSpectra_All, Wavelengths_sorted, MacBethSpectraData, MacBeth_RGB, Name, **kwargs):
	'''
	Function to plot the spectra extracted from the patches of macbeth colour chart
	
	Inputs:
		- PatchesSpectra_All: an array, or a list of arrays. Each array is expected of the shape
			(Nwavelengths (16), Npatches (30), 3). Uses the output of the GetPatchesSpectrum()
			function.
		- Wavelengths_sorted: list of sorted wavelengths
		- MacBethSpectraData: Ground truth spectra for the macbeth patches
		- MacBeth_RGB: MacBeth RBG values for each patch (for plotting)
		- Name: Name of the dataset (for saving)
		
		- kwargs:
			- Help: print this info
			- SavingPath: If indicated, saves the figure at the indicated path
			- ChosenMethod (0). If more than one set of spectra provided, determines which
				of those (the 'method') has the PSNR indicated for each path
			- PlotLabels: What label to put for each provided set of spectra. If not indicated
				a generic 'option 1', 'option 2' etc will be used
			- WhitePatchNormalise (True). Normalises all spectral by the spectra of the white patch

	Outputs:
		- (plots figure)
	
	
	'''
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(PlotPatchesSpectra))
		return

	SavingPath = kwargs.get('SavingPath', '')
	if SavingPath is None:
		SaveFig = False
	else:
		SaveFig = True

	PlotColours = ['limegreen', 'royalblue', 'darkblue', 'orange', 'red', 'cyan', 'magenta']
		
	ChosenMethod = kwargs.get('ChosenMethod', 0)
	WhitePatchNormalise = kwargs.get('WhitePatchNormalise', True)
		
	## If there is only one spectra to plot per patch, place in list to fit in code
	if isinstance(PatchesSpectra_All, list)==False:
		PatchesSpectra_All = [PatchesSpectra_All]
		
	PlotLabels = kwargs.get('PlotLabels')
	if PlotLabels is None:
		print(f'Indicate PlotLabels for more descriptive plot')
		Plotlabels = [f'Option {i}' for i in range(0,len(PatchesSpectra_All))]

	WavelengthRange_start = np.round(int(np.amin(Wavelengths_sorted))/10,0)*10
	WavelengthRange_end = np.round(np.amax(Wavelengths_sorted)/10,0)*10
	print(f'Wavelength range: {WavelengthRange_start} : {WavelengthRange_end}')

	idx_min_gtruth = find_closest(MacBethSpectraData[:,0], WavelengthRange_start)
	idx_max_gtruth = find_closest(MacBethSpectraData[:,0], WavelengthRange_end)

	Nwhite=8-1
	NN = len(Wavelengths_sorted)
	White_truth = MacBethSpectraData[idx_min_gtruth:idx_max_gtruth,Nwhite+1]
	GroundTruthWavelengths = MacBethSpectraData[idx_min_gtruth:idx_max_gtruth,0]

	fig, ax = plt.subplots(nrows=5, ncols=6, figsize=(14,12))
	for i in range(0,6):
		for j in range(0,5):
			patchN = i + j*6
			color=(MacBeth_RGB[patchN,0]/255, MacBeth_RGB[patchN,1]/255, MacBeth_RGB[patchN,2]/255)
			GroundTruthSpectrum = MacBethSpectraData[idx_min_gtruth:idx_max_gtruth,patchN+1]
			if WhitePatchNormalise:
				GroundTruthSpectrumN = np.divide(GroundTruthSpectrum, White_truth)
			else:
				GroundTruthSpectrumN = GroundTruthSpectrum
			ax[j,i].plot(GroundTruthWavelengths, GroundTruthSpectrumN, color='black', lw=4, label='Truth')
			PSNR_Vals = []
			for k in range(0,len(PatchesSpectra_All)):
				PatchesSpectra = PatchesSpectra_All[k]
				if WhitePatchNormalise:
					spectra_WhiteNorm = np.divide(PatchesSpectra[:,patchN,1], PatchesSpectra[:,Nwhite,1])
				else:
					spectra_WhiteNorm = PatchesSpectra[:,patchN,1]
				ax[j,i].plot(Wavelengths_sorted, spectra_WhiteNorm, '.-', c=PlotColours[k], label=PlotLabels[k]) #PlotLinestyles[w], , label=PlotLabels[w]
				GT_comparable = CompareSpectra(Wavelengths_sorted, GroundTruthWavelengths, GroundTruthSpectrumN)

				PSNR = psnr(GT_comparable, spectra_WhiteNorm)
				PSNR_Vals.append(PSNR)

			ax[j,i].set_ylim(-0.05,1.05)
			ax[j,i].set_xlim(450,670)
			
			if len(PSNR_Vals)>1:
				MaxPSNR_pos = np.where(PSNR_Vals==np.amax(PSNR_Vals))[0][0]
			else:
				MaxPSNR_pos = 0
			print(f'Best method for patch {patchN+1} = {PlotLabels[MaxPSNR_pos]}')

			if patchN==Nwhite:
				ax[j,i].set_title(f'Patch {patchN+1} - white', color='black', fontsize=10)# - {itn:.0f} itn,\n {r1norm*10**6:.0f} e-6 r1norm', fontsize=12)
				ax[j,i].legend(fontsize=8, loc='lower center')
			else:
#                 ax[j,i].set_title(f'Patch {patchN+1}\n PSNR = {PSNR:.2f}', color=color, fontsize=10) # fontweight="bold",
				ax[j,i].set_title(f'Patch {patchN+1}\nSelected PSNR = {PSNR_Vals[ChosenMethod]:.2f}\nMax: {np.amax(PSNR_Vals):.2f} {PlotLabels[MaxPSNR_pos]}', 
							  color=color, fontsize=10) # fontweight="bold",

			if j==4:
				ax[j,i].set_xlabel('Wavelength [nm]')
			if j!=4:
				ax[j,i].xaxis.set_ticklabels([])
			if i==0:
				ax[j,i].set_ylabel('Normalized intensity')
			if i!=0:
				ax[j,i].yaxis.set_ticklabels([])

	plt.suptitle(f'Spectra for {Name} - Selected Method: {ChosenMethod}')
	plt.tight_layout()
	if SaveFig:
		plt.savefig(f'{SavingPath}_Patches.png')
	plt.show()













