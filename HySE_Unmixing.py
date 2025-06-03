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
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"




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

	return MixingMatrix
