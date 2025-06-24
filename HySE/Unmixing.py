import numpy as np
# import cv2
# import os
from datetime import datetime
# from scipy.signal import savgol_filter, find_peaks
import matplotlib
from matplotlib import pyplot as plt
# import imageio
# import matplotlib.colors as colors
# import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"


import HySE.Masking



def MakeMixingMatrix(Wavelengths_unsorted, Arduino_MixingMatrix, **kwargs):
	info = '''
	Computes a mixing matrix, based on the Arduino matrix used during data recording
	If FromCalib is set to True, the mixing matrix is computed using the single wavelength
	sweep calibration. Otherwise, the function outputs a binary mixing matrix

	Inputs:
		- Wavelengths_unsorted: Unsorted wavelengths, as input in the NKT software 
		  list with 16 elements, panel 2, panel 4

		- Arduino_MixingMatrix: Arduino mixing matrix

		- kwargs:
			- Help: Print this information message
			- FromCalib = False: If true, use the single wavelength calibration hypercube
				to compute the mixing matrix
			- Hypercube_WhiteCalib: Hypercube computed from the single wavelength calibration
				Required if setting FromCalib to True
			- UseMean = False: Sets the mean value of the calibration instead of the maximal value
				Only used if FromCalib = True
			- Plot=True: Plot mixing matrix
			- SaveFig = False
			- SavingPath = ''

	'''

	Help = kwargs.get('Help', False)
	FromCalib = kwargs.get('FromCalib', False)
	UseMean = kwargs.get('UseMean', False)
	Plot = kwargs.get('Plot', True)
	SaveFig = kwargs.get('SaveFig', False)
	SavingPath = kwargs.get('SavingPath', '')

	if Help:
		print(info)
		return 0

	if FromCalib:
		print('Computing mixing matrix from single wavelength calibration')
		Hypercube_WhiteCalib = kwargs.get('Hypercube_WhiteCalib')
		if Hypercube_WhiteCalib is None:
			print(f'MakeMixingMatrix error:')
			print(f'FromCalib has been set to True. Please input the appropriate Hypercube_WhiteCalib')
			return None  # or raise an Exception
	else:
		print('Computing binary mixing matrix')


	Wavelengths_sorted = np.sort(Wavelengths_unsorted)
	Panel2_wavs = Wavelengths_unsorted[0:8]
	Panel4_wavs = Wavelengths_unsorted[8:]
	MixingMatrix = np.zeros((len(Wavelengths_sorted), len(Wavelengths_sorted)))
	(NN, _) = Arduino_MixingMatrix.shape
	for i in range(0,NN):
		## First profile (profile 2 - REDs)
		bins = Arduino_MixingMatrix[i,:]
		for j in range(0,len(bins)):
			wav_nm = Panel2_wavs[bins[j]]
			wav_k = np.where(Wavelengths_sorted==wav_nm)[0][0]
			if FromCalib:
				if UseMean:
					wav_k_amp = np.nanmean(Hypercube_WhiteCalib[wav_k,:,:])
				else:
					wav_k_amp = np.nanmax(Hypercube_WhiteCalib[wav_k,:,:])
			else:
				wav_k_amp = 1
			MixingMatrix[i,wav_k] = wav_k_amp

		## Second profile (profile 4 - BLUEs)
		for j in range(0,len(bins)):
			wav_nm = Panel4_wavs[bins[j]]
			wav_k = np.where(Wavelengths_sorted==wav_nm)[0][0]
			if FromCalib:
				if UseMean:
					wav_k_amp = np.nanmean(Hypercube_WhiteCalib[wav_k,:,:])
				else:
					wav_k_amp = np.nanmax(Hypercube_WhiteCalib[wav_k,:,:])
			else:
				wav_k_amp = 1
			MixingMatrix[i+8,wav_k] = wav_k_amp

	## Compute and print the matrix determinant
	Matrix_Det = np.linalg.det(MixingMatrix)
	if Matrix_Det==0:
		print(f'PROBLEM! Matrix determinant is 0')
	else:
		print(f'Matrix determinant: {Matrix_Det}')

	if Plot:
		xx = [i for i in range(0,16)]
		bin_labels = [f'im {i+1}' for i in range(0,16)]

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
		ax.imshow(MixingMatrix, cmap='magma')
		ax.set_xticks(xx)
		ax.set_yticks(xx)
		ax.set_xticklabels(Wavelengths_sorted, rotation=90)
		ax.set_yticklabels(bin_labels)

		ax.set_xlabel(f'Individual Wavelengths [nm]')
		ax.set_ylabel(f'Combined Wavelengths [nm]')

		if FromCalib:
			if UseMean:
				ax.set_title(f'Mixing Matrix - From Calibration (mean)')
			else:
				ax.set_title(f'Mixing Matrix - From Calibration (max)')
		else:
			ax.set_title(f'Mixing Matrix - Binary')
		plt.tight_layout()
		plt.show()

	return MixingMatrix



def NormaliseMixedHypercube(MixedHypercube, **kwargs):
	info='''
	Normalises the raw mixed hypercube.
	Dark subtraction and/or white normalisation

	Input:
		- MixedHypercube

		- kwargs:
			- Help
			- Dark = array shape (YY, XX)
			- WhiteCalibration = array shape (Nwavelengths, YY, XX)
			- Sigma = 20. Integer, indicates how much blurring to apply 
				for the dark and White calibration arrays
			- Wavelengths_list = list of wavelengths (unsorted)
				Required for WhiteCalibration
			- Plot = True
			- vmax: float. For plotting range
			- vmin: float. For plotting range
			- SavePlot = False
			- SavingFigure = '' string.
			
	Output:
		- Normalised mixed hypercube. 
			If Dark is indicated, the normalised hypercube will be dark subtracted
			If WhiteCalibration is indicated, the normalised hypercube will be white normalised


	'''
	
	Help = kwargs.get('Help', False)
	if Help:
		print(info)
		
	Sigma = kwargs.get('Sigma', 20)
	Dark = kwargs.get('Dark')
	if Dark is not None:
		Dark_g = gaussian_filter(Dark, sigma=Sigma)
		print(f'Dark subtraction. Avg val = {np.average(Dark):.2f}, after blurring: {np.average(Dark_g):.2f}')
	
	WhiteCalibration = kwargs.get('WhiteCalibration')
	if WhiteCalibration is not None:
		print(f'White Normalising')
		Wavelengths_list = kwargs.get('Wavelengths_list')
		if Wavelengths_list is None:
			print(f'Please indicate Wavelengths_list (unsorted) when requesting WhiteCalibration')
			return 0
		if Dark is not None:
			WhiteCalibration_ = gaussian_filter(np.subtract(WhiteCalibration, Dark_g), sigma=Sigma)
		else:
			WhiteCalibration_ = gaussian_filter(WhiteCalibration, sigma=Sigma)
		
#         wavelength_to_index = {wavelength: idx for idx, wavelength in enumerate(np.sort(Wavelengths_list))}
			
		
			
	Plot = kwargs.get('Plot', True)
	SaveFigure = kwargs.get('SaveFigure', True)
	SavingPath = kwargs.get('SavingPath', '')
	vmax = kwargs.get('vmax', 5)
	vmin = kwargs.get('vmin', 0)
	if (Dark is None) and (WhiteCalibration is None):
		print(f'No normalisation')
		Plot=False
	
	
	if len(MixedHypercube.shape)>3:
		MixedHypercube_ = MixedHypercube
	else:
		MixedHypercube_ = np.array([MixedHypercube])

	if WhiteCalibration is not None:
		Mask = HySE.Masking.GetStandardMask(WhiteCalibration_, threshold=1)
	else:
		print(f'White Calibration not provided. Estimating mask from data itself.')
		Mask = HySE.Masking.GetStandardMask(MixedHypercube_[0], threshold=1)
	print(f'Mask shape: {Mask.shape}')
		
	MixedHypercube_N = np.zeros(MixedHypercube_.shape)
	(SS, WW, YY, XX) = MixedHypercube_.shape
	for s in range(0,SS):
		for w in range(0,WW):
			frame = MixedHypercube_[s,w,:,:]
			if Dark is not None:
				frame_N = np.subtract(frame, Dark_g)
			else:
				frame_N = frame
			if WhiteCalibration is not None:
#                 white_idx = wavelength_to_index[Wavelengths_list[w]]
				white_cal_sub = WhiteCalibration_[w,:,:]
				frame_WN = np.divide(frame_N, white_cal_sub, out=np.zeros_like(frame_N), where=white_cal_sub!=0)
#                 frame_N = np.divide(frame_N.astype('float64'), WhiteCalibration_[white_idx,:,:].astype('float64'), 
#                                     out=np.zeros_like(frame_N), where=WhiteCalibration_[white_idx,:,:]!=0)
#                 frame_N = frame_N
				MixedHypercube_N[s,w,:,:] = np.ma.array(frame_WN, mask=Mask)
			else:
				MixedHypercube_N[s,w,:,:] = np.ma.array(frame_N, mask=Mask)
			
	if Plot:
		HypercubeToPlot = MixedHypercube_N[0,:,:,:]
		nn = 0
		Mavg = np.average(HypercubeToPlot)
		Mstd = np.std(HypercubeToPlot)
#         MM = Mavg+5*Mstd
		fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
		for j in range(0,4):
			for i in range(0,4):
				if nn<17:
					ToPlot_sub = np.ma.array(HypercubeToPlot[nn,:,:], mask=Mask)
					ax[j,i].imshow(ToPlot_sub, cmap='magma',vmin=vmin, vmax=vmax)
					ax[j,i].set_title(f'im {nn}')
					ax[j,i].set_xticks([])
					ax[j,i].set_yticks([])
					nn = nn+1
				else:
					ax[j,i].set_xticks([])
					ax[j,i].set_yticks([])
		plt.tight_layout()
		if SaveFigure:
			plt.savefig(f'{SavingPath}Normalised_Hypercube.png')
		plt.show()
			
	if MixedHypercube_N.shape[0]==1:
		MixedHypercube_N = MixedHypercube_N[0,:,:,:]
			
	return MixedHypercube_N, Mask



def UnmixData(MixedHypercube, MixingMatrix, **kwargs):
	info='''
	Computes the unmixing of the raw hypercube according to a mixing matrix

	Input:
		- MixedHypercube
		- Mixing Matrix

		- kwargs:
			- Help
			- Average = True. If the input mixed hypercube containes more than one shape,
				the function can average the unmixed arrays or leave then individually

	Output:
		- Unmixed hypercube


	'''

	Help = kwargs.get('Help', False)
	Average = kwargs.get('Average', True)
	if Help:
		print(info)

	## Adjust shape of the Hypercube for consistency
	if len(MixedHypercube.shape)>3:
		MixedHypercube_ = MixedHypercube
	else:
		MixedHypercube_ = np.array([MixedHypercube])
		
		
	UnmixedHypercube = []
	SS, WW, YY, XX = MixedHypercube_.shape
	for s in range(0,SS):
		hypercube_sub = MixedHypercube_[s,:,:,:]
		NN, YY, XX = hypercube_sub.shape
		ObservedMatrix = MakeObservedMatrix(hypercube_sub)
#         print(f'MixingMatrix: {MixingMatrix.shape}, ObservedMatrix: {ObservedMatrix}')
		SolvedMatrix_flat = np.linalg.lstsq(MixingMatrix, ObservedMatrix, rcond=-1)[0]
		SolvedMatrix = []
		for n in range(0,NN):
			matrix_sub = SolvedMatrix_flat[n,:].reshape(YY,XX)
			SolvedMatrix.append(matrix_sub)
		UnmixedHypercube.append(SolvedMatrix)
	
	UnmixedHypercube = np.array(UnmixedHypercube)
	if Average:
		UnmixedHypercube = np.average(UnmixedHypercube,axis=0)
	return UnmixedHypercube

		
		
def MakeObservedMatrix(Hypercube):
	N, YY, XX = Hypercube.shape
	Matrix = []
	for n in range(0,N):
		data_sub = Hypercube[n,:,:]
		Matrix.append(data_sub.ravel())
	return np.array(Matrix)
		