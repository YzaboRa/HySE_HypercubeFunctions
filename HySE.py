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



## Functions from HySE_UserTools

def Help():
	HySE_UserTools.Help()

def FindPlottingRange(array):
	mm, MM = HySE_UserTools.FindPlottingRange(array)
	return mm, MM

def wavelength_to_rgb(wavelength, gamma=0.8):
	(R, G, B) = HySE_UserTools.wavelength_to_rgb(wavelength, gamma=0.8)
	return (R, G, B)

def PlotCoRegistered(im_static, im_shifted, im_coregistered, **kwargs):
	HySE_UserTools.PlotCoRegistered(im_static, im_shifted, im_coregistered, **kwargs)

def PlotHypercube(Hypercube, **kwargs):
	HySE_UserTools.PlotHypercube(Hypercube, **kwargs)

def MakeHypercubeVideo(Hypercube, SavingPathWithName, **kwargs):
	HySE_UserTools.MakeHypercubeVideo(Hypercube, SavingPathWithName, **kwargs)



## Functions from HySE_ImportData

def GetSweepData_FromPath(vidPath, EdgePos, Nsweep, **kwargs):
	DataSweep = HySE_ImportData.GetSweepData_FromPath(vidPath, EdgePos, Nsweep, **kwargs)
	return DataSweep

def ImportData(Path, *Coords, **Info):
	data = HySE_ImportData.ImportData(Path, *Coords, **Info)
	return data

def ImportData_imageio(Path, *Coords, **Info):
	data = HySE_ImportData.ImportData_imageio(Path, *Coords, **Info)
	return data



## Functions from HySE_CoRegistrationTools

def GetCoregisteredHypercube(vidPath, EdgePos, Nsweep, Wavelengths_list, **kwargs):
	Hypercube = HySE_CoRegistrationTools.GetCoregisteredHypercube(vidPath, EdgePos, Nsweep, Wavelengths_list, **kwargs)
	return Hypercube

def SweepCoRegister(DataSweep, Wavelengths_list, **kwargs):
	Hypercube_sorted = HySE_CoRegistrationTools.SweepCoRegister(DataSweep, Wavelengths_list, **kwargs)
	return Hypercube_sorted

def CoRegisterImages(im_static, im_shifted):
	im_coregistered, shift_val, time_taken = HySE_CoRegistrationTools.CoRegisterImages(im_static, im_shifted)
	return im_coregistered, shift_val, time_taken



## Functions from HySE_GetHypercubePosition

def FindHypercube(DataPath, Wavelengths_list, **kwargs):
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
	peaks, SGfilter, SGfilter_grad = HySE_GetHypercubePosition.FindPeaks(trace, **kwargs)
	return peaks, SGfilter, SGfilter_grad


def GetPeakDist(peaks, FrameStart, FrameEnd):
	peaks_dist = HySE_GetHypercubePosition.GetPeakDist(peaks, FrameStart, FrameEnd)
	return peaks_dist


def GetEdgesPos(peaks_dist, DarkMin, FrameStart, FrameEnd, MaxPlateauSize, PlateauSize, Ncolours, printInfo=True):
	EdgePos, Stats = HySE_GetHypercubePosition.GetEdgesPos(peaks_dist, DarkMin, FrameStart, FrameEnd, MaxPlateauSize, PlateauSize, Ncolours, printInfo=True)




## Functions from HySE_ManipulateHypercube


def ComputeHypercube(DataPath, EdgePos, Wavelengths_list, **kwargs):
	Hypercube_sorted, Darks = HySE_ManipulateHypercube.ComputeHypercube(DataPath, EdgePos, Wavelengths_list, **kwargs)
	return Hypercube_sorted, Darks


def NormaliseHypercube(DataPath, Hypercube, Hypercube_White, Dark, Wavelengths_list, **kwargs):
	hypercubeN = HySE_ManipulateHypercube.NormaliseHypercube(DataPath, Hypercube, Hypercube_White, Dark, Wavelengths_list, **kwargs)
	return hypercubeN


def Rescale(im, PercMax, Crop=True):
	imrescaled = HySE_ManipulateHypercube.Rescale(im, PercMax, Crop=True)
	return imrescaled


def GetDark(vidPath, EdgePos, **kwargs):
	DarkAvg = HySE_ManipulateHypercube.GetDark(vidPath, EdgePos, **kwargs)
	return DarkAvg

def GetDark_FromData(DataAll, EdgePos, **kwargs):
	DarkAvg = HySE_ManipulateHypercube.GetDark(DataAll, EdgePos, **kwargs)
	return DarkAvg



























