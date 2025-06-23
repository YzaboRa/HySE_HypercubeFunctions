"""

Moduel to import. 

The functions have been spit in different files for improved readability and manageability.

To avoid having to import all files individually and to keep track of where each function is located,
this code is designed to act as an interface to faciliate user import.



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

import HySE_UserTools
import HySE_ImportData
import HySE_CoRegistrationTools
import HySE_GetHypercubePosition
import HySE_ManipulateHypercube
import HySE_Mask
import HySE_Unmixing



## Functions from HySE_UserTools

def Help():
	HySE_UserTools.Help()

def FindPlottingRange(array):
	mm, MM = HySE_UserTools.FindPlottingRange(array)
	return mm, MM

def find_closest(arr, val):
	idx = HySE_UserTools.find_closest(arr, val)
	return idx

def wavelength_to_rgb(wavelength, gamma=0.8):
	(R, G, B) = HySE_UserTools.wavelength_to_rgb(wavelength, gamma=0.8)
	return (R, G, B)

def PlotCoRegistered(im_static, im_shifted, im_coregistered, **kwargs):
	"""
	kwargs = ShowPlot (False), SavePlot (False), SavingPathWithName ('')
	"""
	HySE_UserTools.PlotCoRegistered(im_static, im_shifted, im_coregistered, **kwargs)

def PlotHypercube(Hypercube, **kwargs):
	"""
	kwargs = Wavelengths, Masks, SavePlot (False), SavingPathWithName (''), ShowPlot (True), SameScale (False), Help
	"""
	HySE_UserTools.PlotHypercube(Hypercube, **kwargs)

def MakeHypercubeVideo(Hypercube, SavingPathWithName, **kwargs):
	"""
	kwargs = fps (10)
	"""
	HySE_UserTools.MakeHypercubeVideo(Hypercube, SavingPathWithName, **kwargs)

def PlotDark(Dark):
	"""
	Dark: 2D numpy array
	"""
	HySE_UserTools.PlotDark(Dark)


## For Macbeth colourcharts
def GetPatchPos(Patch1_pos, Patch_size_x, Patch_size_y, Image_angle):
	Positions = HySE_UserTools.GetPatchPos(Patch1_pos, Patch_size_x, Patch_size_y, Image_angle)
	return Positions

def GetPatchesSpectrum(Hypercube, Sample_size, Positions, CropCoordinates):
	Spectrum = HySE_UserTools.GetPatchesSpectrum(Hypercube, Sample_size, Positions, CropCoordinates)
	return Spectrum

def GetPatchesIntensity(Image, Sample_size, PatchesPositions):
	Intensities = HySE_UserTools.GetPatchesIntensity(Image, Sample_size, PatchesPositions)
	return Intensities

def PlotPatchesDetection(Macbeth, Positions, Sample_size):
	HySE_UserTools.PlotPatchesDetection(Macbeth, Positions, Sample_size)

def psnr(img1, img2):
	psnr = HySE_UserTools.psnr(img1, img2)
	return psnr(img1, img2)

def CompareSpectra(Wavelengths_sorted, GroundTruthWavelengths, GroundTruthSpectrum):
	ComparableSpectra = HySE_UserTools.CompareSpectra(Wavelengths_sorted, GroundTruthWavelengths, GroundTruthSpectrum)
	return ComparableSpectra

def PlotPatchesSpectra(PatchesSpectra_All, Wavelengths_sorted, MacBethSpectraData, MacBeth_RGB, Name, **kwargs):
	HySE_UserTools.PlotPatchesSpectra(PatchesSpectra_All, Wavelengths_sorted, MacBethSpectraData, MacBeth_RGB, Name, **kwargs)



## Functions from HySE_ImportData

def GetSweepData_FromPath(vidPath, EdgePos, Nsweep, **kwargs):
	"""
	kwargs = CropImDimensions (CCRC HD video)
	"""
	DataSweep = HySE_ImportData.GetSweepData_FromPath(vidPath, EdgePos, Nsweep, **kwargs)
	return DataSweep

def ImportData(Path, *Coords, **kwargs):
	"""
	kwargs = RGB (False), Trace (False), CropIm (True), CropImDimensions (CCRC HD video)
	"""
	data = HySE_ImportData.ImportData(Path, *Coords, **kwargs)
	return data

def ImportData_imageio(Path, *Coords, **kwargs):
	"""
	kwargs = RGB (False), Trace (False), CropIm (True), CropImDimensions (CCRC HD video)
	"""
	data = HySE_ImportData.ImportData_imageio(Path, *Coords, **kwargs)
	return data



## Functions from HySE_GetHypercubePosition

def FindHypercube(DataPath, Wavelengths_list, **kwargs):
	"""
	kwargs = 
		- Help = True: to print help message 
		- PlotGradient = True: To plot gratient of smoothed trace and detected peaks 
			To see effect of other parameters when optimising 
		- PrintPeaks = True: To print the list of all detected peaks and their positions 
		- MaxPlateauSize = Integer: Set the maximal expected size for a plateau. 
		- WindowLength = Integer: Window over which the smoothing of the trace is performed 
			If the data consists of NxRGB cycles, this number should be a factor of 3 
		- PolyOrder = Integer: Order of the polynomial used in smoothing (Savitzky-Golay) 
		- PeakHeight = Float: Detection threshold applied to the gradient of the smoothed trace 
			to find edges between neighbouring colours 
		- PeakDistance = Integer: Minimal distance between neightbouring peaks/plateaux 
			Depends on the repeat number, and will impact the detection of double plateaux 
		- DarkMin = Integer: Set the minimal size of the long dark between succesive sweeps 
			Depends on the repeat numbner, and will impact the detection of individial sweeps 
		- PlateauSize = Integer: Set the expected average size for a plateau (in frame number) 
			Depends on the repeat number and will impact how well double plateaux are handled 
			Automatically adjusts expected size when plateaux are detected, but needs to be set 
			manually if a full sweep could not be detected automatically. 
		- CropImDimensions = [xmin, xmax, ymin, ymax]: coordinates of image crop (default Full HD) 
		- ReturnPeaks = True: if want the list of peaks and peak distances 
			(for manual tests, for example if fewer than 8 colours 
		- Ncolours = integer: if different from 8 (for example, if one FSK was off) 
		- SaveFig = True
	"""
	try:
		ReturnPeaks = kwargs['ReturnPeaks']
	except KeyError:
		ReturnPeaks = False
	
	if ReturnPeaks:
		EdgePos, peaks, peaks_dist = HySE_GetHypercubePosition.FindHypercube(DataPath, Wavelengths_list, **kwargs)
		return EdgePos, peaks, peaks_dist
	else: 
		EdgePos = HySE_GetHypercubePosition.FindHypercube(DataPath, Wavelengths_list, **kwargs)
		return EdgePos

def FindPeaks(trace, **kwargs):
	"""
	kwargs: window_length (6), polyorder (1), peak_height (0.03), peak_distance (14)
	"""
	peaks, SGfilter, SGfilter_grad = HySE_GetHypercubePosition.FindPeaks(trace, **kwargs)
	return peaks, SGfilter, SGfilter_grad


def GetPeakDist(peaks, FrameStart, FrameEnd):
	peaks_dist = HySE_GetHypercubePosition.GetPeakDist(peaks, FrameStart, FrameEnd)
	return peaks_dist


def GetEdgesPos(peaks_dist, DarkMin, FrameStart, FrameEnd, MaxPlateauSize, PlateauSize, Ncolours, printInfo=True):
	EdgePos, Stats = HySE_GetHypercubePosition.GetEdgesPos(peaks_dist, DarkMin, FrameStart, FrameEnd, MaxPlateauSize, PlateauSize, Ncolours, printInfo=True)




## Functions from HySE_ManipulateHypercube

def ComputeHypercube(DataPath, EdgePos, Wavelengths_list, **kwargs):
	"""
	kwargs = Help, BufferSize (10), Name (''), SaveFig, SaveArray
	"""
	Hypercube_sorted, Darks = HySE_ManipulateHypercube.ComputeHypercube(DataPath, EdgePos, Wavelengths_list, **kwargs)
	return Hypercube_sorted, Darks


def NormaliseHypercube(DataPath, Hypercube, Hypercube_White, Dark, Wavelengths_list, **kwargs):
	"""
	kwargs = Name ('')
	"""
	hypercubeN = HySE_ManipulateHypercube.NormaliseHypercube(DataPath, Hypercube, Hypercube_White, Dark, Wavelengths_list, **kwargs)
	return hypercubeN


def Rescale(im, PercMax, Crop=True):
	imrescaled = HySE_ManipulateHypercube.Rescale(im, PercMax, Crop=True)
	return imrescaled

def GetLongDark(vidPath, EdgePos, **kwargs):
	"""
	kwargs: Help, ExtraWav (0), Buffer (20)
	"""
	LongDark = HySE_ManipulateHypercube.GetLongDark(vidPath, EdgePos, **kwargs)
	return LongDark

# def GetDark(vidPath, EdgePos, **kwargs):
# 	"""
# 	kwargs = CropImDimensions (CCRC HD video), Buffer (6), DarkRepeat (3), SaveDark (True), SavePath ('')
# 	"""
# 	DarkAvg = HySE_ManipulateHypercube.GetDark(vidPath, EdgePos, **kwargs)
# 	return DarkAvg

# def GetDark_FromData(DataAll, EdgePos, **kwargs):
# 	"""
# 	kwargs = CropImDimensions (CCRC HD video), Buffer (6), DarkRepeat (3), SaveDark (True), SavePath ('')
# 	"""
# 	DarkAvg = HySE_ManipulateHypercube.GetDark(DataAll, EdgePos, **kwargs)
	# return DarkAvg

def NormaliseFrames(image, image_white, image_dark):
	imageN = HySE_ManipulateHypercube.NormaliseFrames(image, image_white, image_dark)
	return imageN



## Functions from HySE_CoRegistrationTools

def SweepRollingCoRegister_WithNormalisation(DataSweep, WhiteHypercube, Dark, Wavelengths_list, **kwargs):
	"""
	kwargs = Buffer (6), ImStatic_Wavelength (550), ImStatic_Index (8), PlotDiff (False), SavingPath (''), Plot_PlateauList ([5])
			 Plot_Index (14), SaveHypercube (False), Help (False)
	"""
	Hypercube = HySE_CoRegistrationTools.SweepRollingCoRegister_WithNormalisation(DataSweep, WhiteHypercube, Dark, Wavelengths_list, **kwargs)
	return Hypercube

def GetCoregisteredHypercube(vidPath, EdgePos, Nsweep, Wavelengths_list, **kwargs):
	"""
	kwargs = CropImDimensions (CCRC HD video), Buffer (6), ImStatic_Plateau (1), ImStatic_Index (8), SaveHypercube (True)
			 PlotDiff (False), SavingPath (''), Plot_PlateauList ('All'), Plot_Index (14)
	"""
	Hypercube = HySE_CoRegistrationTools.GetCoregisteredHypercube(vidPath, EdgePos, Nsweep, Wavelengths_list, **kwargs)
	return Hypercube

def SweepCoRegister_WithNormalisation(DataSweep, WhiteHypercube, Dark, Wavelengths_list, **kwargs):
	"""
	kwargs = Buffer (6), ImStatic_Plateau (1), ImStatic_Index (8), SaveHypercube (True)
			 PlotDiff (False), SavingPath (''), Plot_PlateauList ('All'), Plot_Index (14)
	"""
	HypercubeNormalised = HySE_CoRegistrationTools.SweepCoRegister_WithNormalisation(DataSweep, WhiteHypercube, Dark, Wavelengths_list, **kwargs)
	return HypercubeNormalised

def SweepCoRegister(DataSweep, Wavelengths_list, **kwargs):
	"""
	kwargs = Buffer (6), ImStatic_Plateau (1), ImStatic_Index (8), SaveHypercube (True)
			 PlotDiff (False), SavingPath (''), Plot_PlateauList ('All'), Plot_Index (14)
	"""
	Hypercube_sorted = HySE_CoRegistrationTools.SweepCoRegister(DataSweep, Wavelengths_list, **kwargs)
	return Hypercube_sorted

def CoRegisterImages(im_static, im_shifted, **kwargs):
	"""
	kwargs = Affine (False)
	"""
	im_coregistered, shift_val, time_taken = HySE_CoRegistrationTools.CoRegisterImages(im_static, im_shifted, **kwargs)
	return im_coregistered, shift_val, time_taken



## Functions from HySE_Mask

def GetStandardMask(WhiteCalibration, **kwargs):
	Mask = HySE_Mask.GetStandardMask(WhiteCalibration, **kwargs)
	return Mask

def ConvertMaskToBinary(mask):
	binary_mask = HySE_Mask.ConvertMaskToBinary(mask)
	return binary_mask

def BooleanMaskOperation(bool_white, bool_wav):
	bool_result = HySE_Mask.BooleanMaskOperation(bool_white, bool_wav)
	return bool_result

def TakeWavMaskDiff(mask_white, mask_shifted):
	result = HySE_Mask.TakeWavMaskDiff(mask_white, mask_shifted)
	return result

def CombineMasks(mask_white, mask_shifted):
	mask = HySE_Mask.CombineMasks(mask_white, mask_shifted)
	return mask

def GetMask(frame, **kwargs):
	"""
	kwargs: LowCutoff, HighCutoff, PlotMask, Help
	"""
	combined_mask = HySE_Mask.GetMask(frame, **kwargs)
	return combined_mask

def CoRegisterImages_WithMask(im_static, im_moving, **kwargs):
	"""
	kwargs: StaticMask, MovingMask, Affine, Verbose, Help
	"""
	im_coregistered = HySE_Mask.CoRegisterImages_WithMask(im_static, im_moving, **kwargs)
	return im_coregistered

def SweepCoRegister_MaskedWithNormalisation(DataSweep, WhiteHypercube, Dark, Wavelengths_list, **kwargs):
	"""
	kwargs: Buffer (6), ImStatic_Wavelength (550), ImStatic_Index (8), LowCutoff (False), HighCutoff (False), Mask_CombinedAvgCutoff (0.01),
	SavingPath (''), SaveHypercube (True), PlotDiff (False), Plot_PlateauList (5, 'All'/'None'), Plot_Index (14), Help
	"""
	Hypercube, hypercube_masks = HySE_Mask.SweepCoRegister_MaskedWithNormalisation(DataSweep, WhiteHypercube, Dark, Wavelengths_list, **kwargs)
	return Hypercube, hypercube_masks


def SweepRollingCoRegister_MaskedWithNormalisation(DataSweep, WhiteHypercube, Dark, Wavelengths_list, **kwargs):
	"""
	kwargs: Buffer (6), ImStatic_Wavelength (550), ImStatic_Index (8), LowCutoff (False), HighCutoff (False), Mask_CombinedAvgCutoff (0.01),
	SavingPath (''), SaveHypercube (True), PlotDiff (False), Plot_PlateauList (5, 'All'/'None'), Plot_Index (14), Help
	"""
	Hypercube, hypercube_masks = HySE_Mask.SweepRollingCoRegister_MaskedWithNormalisation(DataSweep, WhiteHypercube, Dark, Wavelengths_list, **kwargs)
	return Hypercube, hypercube_masks




## Functions from HySE_Unmixing

def MakeMixingMatrix(Wavelengths_unsorted, Arduino_MixingMatrix, **kwargs):
	"""
	kwargs: Help, FromCalib (False), Hypercube_WhiteCalib, UseMean (False), Plot (True), SaveFig (False), SavingPath ('')
	"""
	MixingMatrix = HySE_Unmixing.MakeMixingMatrix(Wavelengths_unsorted, Arduino_MixingMatrix, **kwargs)
	return MixingMatrix

def NormaliseMixedHypercube(MixedHypercube, **kwargs):
	"""
	kwargs: Help, Dark, WhiteCalibration (*HySE calibration), Sigma, Wavelengths, Plot (True), vmax, vmin, SavePlot (False), SavingFigure ('')
	"""
	MixedHypercube_N, Mask = HySE_Unmixing.NormaliseMixedHypercube(MixedHypercube, **kwargs)
	return MixedHypercube_N, Mask

def UnmixData(MixedHypercube, MixingMatrix, **kwargs):
	"""
	kwargs: Help, Average (True)
	"""
	UnmixedHypercube = HySE_Unmixing.UnmixData(MixedHypercube, MixingMatrix, **kwargs)
	return UnmixedHypercube

def MakeObservedMatrix(Hypercube):
	"""
	kwargs:
	"""
	Matrix = HySE_Unmixing.MakeObservedMatrix(Hypercube)
	return Matrix



















