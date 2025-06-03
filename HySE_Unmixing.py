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




def MakeMixingMatrix(Wavelengths_sorted, Arduino_MixingMatrix):
	## Outputs binary mixing matrix
	MixingMatrix = np.zeros((len(Wavelengths_sorted), len(Wavelengths_sorted)))
	(NN, _) = Arduino_MixingMatrix.shape
	for i in range(0,NN):
		## First profile (profile 2 - REDs)
		bins = Arduino_MixingMatrix[i,:]
		for j in range(0,len(bins)):
			wav_nm = Panel2_wavs[bins[j]]
			wav_k = np.where(Wavelengths_sorted==wav_nm)[0][0]
			MixingMatrix[i,wav_k] = 1
			
		## Second profile (profile 4 - BLUEs)
		for j in range(0,len(bins)):
			wav_nm = Panel4_wavs[bins[j]]
			wav_k = np.where(Wavelengths_sorted==wav_nm)[0][0]
			MixingMatrix[i+8,wav_k] = 1

	return MixingMatrix



def MakeMixingMatrix_FromCalib_max(Hypercube_WhiteCalib, Wavelengths_sorted, Arduino_MixingMatrix):
	## Outputs float mixing matrix, based on single wavelength sweep calibration
	MixingMatrix = np.zeros((len(Wavelengths_sorted), len(Wavelengths_sorted)))
	(NN, _) = Arduino_MixingMatrix.shape
	for i in range(0,NN):
		
		## First profile (profile 2 - REDs)
		bins = Arduino_MixingMatrix[i,:]
		for j in range(0,len(bins)):
			wav_nm = Panel2_wavs[bins[j]]
			wav_k = np.where(Wavelengths_sorted==wav_nm)[0][0]
			wav_k_amp = np.amax(Hypercube_WhiteCalib[wav_k,:,:])
			MixingMatrix[i,wav_k] = wav_k_amp
			
		## Second profile (profile 4 - BLUEs)
		for j in range(0,len(bins)):
			wav_nm = Panel4_wavs[bins[j]]
			wav_k = np.where(Wavelengths_sorted==wav_nm)[0][0]
			wav_k_amp = np.amax(Hypercube_WhiteCalib[wav_k,:,:])
			MixingMatrix[i+8,wav_k] = wav_k_amp

	return MixingMatrix