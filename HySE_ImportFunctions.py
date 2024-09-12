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
	HySE_UserTools.FindPlottingRange(array)

def wavelength_to_rgb(wavelength, gamma=0.8):
	HySE_UserTools.wavelength_to_rgb(wavelength, gamma=0.8)

def PlotCoRegistered(im_static, im_shifted, im_coregistered, **kwargs):
	HySE_UserTools.PlotCoRegistered(im_static, im_shifted, im_coregistered, **kwargs)

def PlotHypercube(Hypercube, **kwargs):
	HySE_UserTools.PlotHypercube(Hypercube, **kwargs)

def MakeHypercubeVideo(Hypercube, SavingPathWithName, **kwargs):
	HySE_UserTools.MakeHypercubeVideo(Hypercube, SavingPathWithName, **kwargs)



## Functions from HySE_ImportData

def ImportData(Path, *Coords, **Info):
	HySE_ImportData.ImportData(Path, *Coords, **Info)

def ImportData_imageio(Path, *Coords, **Info):
	HySE_ImportData.ImportData_imageio(Path, *Coords, **Info)



## Functions from HySE_CoRegistrationTools

def GetSweepData_FromPath(vidPath, EdgePos, Nsweep, **kwargs):
	HySE_CoRegistrationTools.GetSweepData_FromPath(vidPath, EdgePos, Nsweep, **kwargs)


def CoRegisterImages(im_static, im_shifted):
	HySE_CoRegistrationTools.CoRegisterImages(im_static, im_shifted)



## Functions from HySE_GetHypercubePosition

def FindHypercube(DataPath, Wavelengths_list, **kwargs):
	HySE_GetHypercubePosition.FindHypercube(DataPath, Wavelengths_list, **kwargs)

def FindPeaks(trace, **kwargs):
	HySE_GetHypercubePosition.FindPeaks(trace, **kwargs)


def GetPeakDist(peaks, FrameStart, FrameEnd):
	HySE_GetHypercubePosition.GetPeakDist(peaks, FrameStart, FrameEnd)


def GetEdgesPos(peaks_dist, DarkMin, FrameStart, FrameEnd, MaxPlateauSize, PlateauSize, Ncolours, printInfo=True):
	HySE_GetHypercubePosition.GetEdgesPos(peaks_dist, DarkMin, FrameStart, FrameEnd, MaxPlateauSize, PlateauSize, Ncolours, printInfo=True)




## Functions from HySE_ManipulateHypercube

def SweepCoRegister(DataSweep, Wavelengths_list, **kwargs):
	HySE_ManipulateHypercube.SweepCoRegister(DataSweep, Wavelengths_list, **kwargs):


def ComputeHypercube(DataPath, EdgePos, Wavelengths_list, **kwargs):
	HySE_ManipulateHypercube.ComputeHypercube(DataPath, EdgePos, Wavelengths_list, **kwargs)


def NormaliseHypercube(DataPath, Hypercube, Hypercube_White, Dark, Wavelengths_list, **kwargs):
	HySE_ManipulateHypercube.NormaliseHypercube(DataPath, Hypercube, Hypercube_White, Dark, Wavelengths_list, **kwargs)


def Rescale(im, PercMax, Crop=True):
	HySE_ManipulateHypercube.Rescale(im, PercMax, Crop=True)


def GetDark(DataAll, EdgePos, **kwargs):
	HySE_ManipulateHypercube.GetDark(DataAll, EdgePos, **kwargs)



























