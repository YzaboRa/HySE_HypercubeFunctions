"""

Functions used for wavelenght unmixing

"""

import numpy as np
from datetime import datetime
import matplotlib
from matplotlib import pyplot as plt
import inspect
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"


import HySE.Masking



def MakeMixingMatrix(Wavelengths_unsorted, Arduino_MixingMatrix, **kwargs):
	'''
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

	Outputs:
		- MixingMatrix

	'''

	Help = kwargs.get('Help', False)
	FromCalib = kwargs.get('FromCalib', False)
	UseMean = kwargs.get('UseMean', False)
	Plot = kwargs.get('Plot', True)
	SaveFig = kwargs.get('SaveFig', False)
	SavingPath = kwargs.get('SavingPath', '')

	if Help:
		# print(info)
		print(inspect.getdoc(MakeMixingMatrix))
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

def BlurWhiteCalibration(WhiteCalibration, Sigma):
	"""
	Function that applies Gaussian blurring to every frame in the white reference

	Inputs:
		- WhiteCalibration
		- Sigma (blurring)

	Outputs:
		- WhiteCalibrationBlurred

	"""
	(WW, YY, XX) = WhiteCalibration.shape
	WhiteCalibrationBlurred = np.zeros(WhiteCalibration.shape)
	for w in range(0,WW):
		cal_sub = WhiteCalibration[w,:,:]
		cal_sub_blurred = gaussian_filter(cal_sub, sigma=Sigma)
		WhiteCalibrationBlurred[w,:,:] = cal_sub_blurred
	return WhiteCalibrationBlurred

def SubtractDark(WhiteCalibration, Dark_g):
	"""
	Function that subtacts dark from every frame in hypecube

	Inputs:
		- WhiteCalibration
		- Dark_g

	Outputs:
		- WhiteCalibration_D

	"""
	(WW, YY, XX) = WhiteCalibration.shape
	WhiteCalibration_D = np.zeros(WhiteCalibration.shape)
	for w in range(0,WW):
		cal_sub = WhiteCalibration[w,:,:]
		cal_sub_d = np.subtract(cal_sub, Dark_g)
		WhiteCalibration_D[w,:,:] = cal_sub_d
	return WhiteCalibration_D



def NormaliseMixedHypercube(MixedHypercube, **kwargs):
	'''
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
			- SpectralNormalisation = False. If True, the white reference will be used to normalise the spectrum
				instead of both spectrally and spatially.
			- Wavelengths_list = list of wavelengths (unsorted)
				Required for WhiteCalibration
			- vmax: float. For plotting range
			- vmin: float. For plotting range
			- SavePlot = False
			- SavingFigure = '' string.
			- Plot = True
			
		- Normalised mixed hypercube. 
			If Dark is indicated, the normalised hypercube will be dark subtracted
			If WhiteCalibration is indicated, the normalised hypercube will be white normalised


	Outputs:
		- Normalised Hypercube

		- Mask
			Removes black corners from the Olympus display. Estimated from the white calibration if provided,
			otherwise it uses the data itself to estimate the mask

	'''

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(NormaliseMixedHypercube))
		return 0,0

	Sigma = kwargs.get('Sigma', 20)
	Dark = kwargs.get('Dark')
	if Dark is not None:
		Dark_g = gaussian_filter(Dark, sigma=Sigma)
		print(f'Dark subtraction. Avg val = {np.average(Dark):.2f}, after blurring: {np.average(Dark_g):.2f}')

	SpectralNormalisation = kwargs.get('SpectralNormalisation', False)

	WhiteCalibration = kwargs.get('WhiteCalibration')
	if WhiteCalibration is not None:
		print(f'White Normalising')
		Wavelengths_list = kwargs.get('Wavelengths_list')
		if Wavelengths_list is None:
			print(f'Please indicate Wavelengths_list (unsorted) when requesting WhiteCalibration')
			return 0
		if Dark is not None:
			WhiteCalibration_ = BlurWhiteCalibration(SubtractDark(WhiteCalibration, Dark_g), Sigma)
			print(f'WC-Dark: {np.average(np.average(WhiteCalibration_, axis=1), axis=1)}')
		else:
			WhiteCalibration_ = BlurWhiteCalibration(WhiteCalibration, Sigma)
			print(f'WC: {np.average(np.average(WhiteCalibration_, axis=1), axis=1)}')

	if (SpectralNormalisation and WhiteCalibration is None):
				print(f'Please input WhiteCalibration when requesting SpectralNormalisation')
				return 0

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

	MixedHypercube_N = np.zeros(MixedHypercube_.shape)
	(SS, WW, YY, XX) = MixedHypercube_.shape
	for s in range(0,SS):
		MixedHypercube_sub = MixedHypercube_[s,:,:,:]
		if Dark is not None:
			MixedHypercube_subN = SubtractDark(MixedHypercube_sub, Dark_g)
		else:
			MixedHypercube_subN = MixedHypercube_sub
			
		for w in range(0,WW):
			frame = MixedHypercube_subN[w,:,:]
			if WhiteCalibration is None:
				frameN = frame
			else:
				whiteframe = WhiteCalibration_[w,:,:]
				if SpectralNormalisation:
					whiteval = np.average(whiteframe)
					frameN = frame/whiteval
				else:
					frameN = np.divide(frame, whiteframe, out=np.zeros_like(frame), where=whiteframe!=0)
			MixedHypercube_N[s,w,:,:] = np.ma.array(frameN, mask=Mask)

	if Plot:
		HypercubeToPlot = MixedHypercube_N[0,:,:,:]
		nn = 0
		Mavg = np.average(HypercubeToPlot)
		Mstd = np.std(HypercubeToPlot)
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
	'''
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
		# print(info)
		print(inspect.getdoc(UnmixData))
		return 0

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
	"""
	Make observed matrix (ravel) from the hypercube to feed in the unmixing algorithm

	Inputs:
		- Hypercube

	Outptus:
		- ObservedMatrix


	"""
	N, YY, XX = Hypercube.shape
	Matrix = []
	for n in range(0,N):
		data_sub = Hypercube[n,:,:]
		Matrix.append(data_sub.ravel())
	return np.array(Matrix)
		