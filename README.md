# HySE_HypercubeFunctions
Functions to compute hypercube from endoscope videos 

## Installation
This is a python 3 code, it requires the following packages:

- numpy
- opencv-python
- DateTime
- scipy
- matplotlib
- ffmpeg
- natsort
- PIL
- imageio
- tqdm
- inspect
- math
- glob

The co-registration is done with SimpleITK.

Documentation available here:
https://simpleitk.org/

SimpleITK can be installed with
```python
pip install SimpleITK
```
N.B. A previous version required SimpleElastix, which cannot be installed properly - Only SimpleITK is now required

It also requires those additional libraries:
- time
- tqdm

If SimpleITK is not installed, or cannot be imported, the code will simply ignore it. This will not be a problem as long as the co-registration functions are not used.


## General information
Last updated: 29 January 2026
Includes handling of 3 wavelengths split unmixing with red reference channel.
Does NOT include actual normalisation using the red references channel, to be added once finalised.

### Help
Every function in the HySE library includes documentation, which can be accessed using the Help function.
A general help function allows to print all the modules and associated functions:
```python
HySE.help()
```

Adding a module in argument allows to print in more details the functinos it contains:
```python
HySE.help('Unmixing')
```

A detailed description of a given function with its inputs and outputs can be accessed by adding the function to the argument:
```python
HySE.help('Unmixing.MakeMixingMatrix')
```

Most functions also allow the input of 'Help=True' as an argument, which will print the same detailed function description. 
The output (0) will match the size of the expected output in order to allow debugging on the go.

```python
HySE.MakeMixingMatrix(_,_,Help=True)
```

## Pre-Processing

### Import 
The raw data needs to be imported and registered. 
Manual registration is done interactively with a QT GUI. To enable this feature in a jupyter notebook, ensure that the right matplotlib feature is set:
```python
%matplotlib qt
```

Import the HySE functions from wherever they are located:
```python
PathToFunctions = 'GitHub/HySE_HypercubeFunctions/'
sys.path.append(PathToFunctions)
import HySE
```

Define where the data is located. Note that a "Name" is defined in order to be added when saving outputs. This helps identify what each output is for.
```python
SavingPath = '/SavingPath/'

DataPath_General = SavingPath+'Videos/'
DarkPath = DataPath_General+'2025-11-25_08-25-28_Dark.mp4'

DataPath = DataPath_General+'2025-11-25_09-17-27_NegControl.mp4'
Name = 'NegControl'
RegistrationSavingPath = SavingPath+'Results/Registered/'

```

### Load Dark
This step only requires the dark video. If it doesn't exist, a dark estimate can be generated from the data itself.
```python
LongDark = HySE.GetDark_WholeVideo(DarkPath) #, CropImDimensions=CropImDimensions
HySE.PlotDark(LongDark)
```

### Load Trace
The following parameters are set for the current implementation. The automatic edge detection is enabled. Make sure that the edges have been properly identified.
```python
EdgePos_Data = HySE.FindHypercube_RGB(DataPath, Automatic=True, PlateauSize=9, DarkMin=12, MaxPlateauSize=20, SaveFig=False)
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/7fd87cd3-ab97-457a-9f21-80580b51f00e" width="800"/>
</p>

### Extract Frames
This uses the edge positions found with the trace to extract usable frames. 
The buffer sets how many frames a thrown out at the start end the end of the plateau. Only the middle frames are used, where the illumination and Olympus post-processing is more stable.
```python
Buffer = 3
Frames, RGB_Dark = HySE.ComputeHypercube_RGB(DataPath, EdgePos_Data, Buffer=Buffer, BlueShift=1, SaveArray=False)
```

### Dark Subtraction
Subtract the dark from all the frames
(Not necessary for registration)
```python
## Data
(Nsweeps, Nwav, Nframes, Y, X, _) = Frames.shape

## BGR 
Frames_Blue = Frames[:,:,:,:,:,0]
Frames_Green = Frames[:,:,:,:,:,1]
Frames_Red = Frames[:,:,:,:,:,2]

# NormaliseMixedHypercube
Frames_BlueD, MaskB = HySE.NormaliseMixedHypercube(Frames_Blue, Dark=LongDark, SaveFigure=False, Plot=False)
Frames_GreenD, MaskG = HySE.NormaliseMixedHypercube(Frames_Green, Dark=LongDark, SaveFigure=False, Plot=False)
Frames_RedD, MaskR = HySE.NormaliseMixedHypercube(Frames_Red, Dark=LongDark, SaveFigure=False, Plot=False)

print(Frames_GreenD.shape)
```

### Save Frames to Identify Good Sweeps
This step must only be done once and is best kept commented out the rest of the time.
This function will create folders for each full sweep and save the frames for each wavelength/combination of wavelength, for each sweep. 
This is done so that the user can manually identify which sweeps contain usable frames for the full sweep and to target registration on the best sweeps.
There are two options.
Option A saves every every frame as png images:
```python
RawFrames_SavingPath = SavingPath
NameSub = ''
HySE.SaveFramesSweeps(Frames_RedD, RawFrames_SavingPath, Name, NameSub, Video=False, Frames=[1], RescalePercentile=99) 
```
And option saves a mp4 video with of each frame in the sweep:
```python
RawFrames_SavingPath = SavingPath
NameSub = ''
HySE.SaveFramesSweeps(Frames_RedD, RawFrames_SavingPath, Name, NameSub, Video=True, Frames=[1], RescalePercentile=99)
```
The "Frames" argument can be used to save more or fewer frames per plateau. Frames=[0,1,2] will save all 3 frames.


### Mask
```python
_, Mask = HySE.NormaliseMixedHypercube(Frames_GreenD[0,:,0,:,:], Dark=LongDark, Wavelengths_list=Wavelengths_list, 
                                       SaveFigure=False, SavingPath=SavingPath+Name, vmax=160, Plot=False)

EdgeMask = HySE.GetBestEdgeMask(Mask)
```

### Register
#### Select Frames to Register
We can only register specific sweeps at a time.
```python
Nsweep = 4
Nframe = 1

# If taking single frames
HypercubeForRegistration = HySE.GetHypercubeForRegistration(Nsweep, Nframe, DataPath, EdgePos_Data, Wavelengths_list, Buffer=Buffer)
print(HypercubeForRegistration.shape)
```

#### Manual Registration
Manual registration hinges on fixed points in each images that must be manually annotated. The code will create a surface (thin plate spline) that goes through all the points. This code was written with Gemini.
Previously defined points can be pre-loaded
```python
ManualRegistrationPointsPath = '/PATH_TO_POINTS/' ## npz file
AllPoints_Loaded = np.load(ManualRegistrationPointsPath, allow_pickle=True)['arr_0'].item()
```

Otherwise points can be defined here. 
Running this function will prompt a window to appear, requesting the user to define points. The code will first prompt the user to define points on the fixed frame, before then requesting the user to define those same points in every other frame to be registered. 
There is no limit to the number of points defined, but point one must be present in all the frames.
A series of features are available to make this step slightly easier. Pressing "z" will undo the last point defined. Pressing "p" will toggle the labels on and off. Use this to make sure that the points are set in the same order each time, to avoid issues.
The colormap, min and max levels of the image can be adjusted. Normal matplotlib features remain available (zoom, saving, etc.)

Another function (CoRegisterHypercubeAndMask) allows the user to set specific ROIs, but uses the automatic registration method on that area (see below).

```python
UserDefinedROI = False
# Cropping = 0
# GridSpacing = 100 # for automatix
index = 0 # which frame is the static one

AllLandmarkPoints = None
# AllLandmarkPoints = AllPoints_Loaded

if Manual:
    if AllLandmarkPoints is not None:
        print(f'Doing manual registration - Using provided fixed points')
        CoregisteredHypercube, AllTransforms, CombinedMask, AllPoints = HySE.CoRegisterHypercubeAndMask_Manual(HypercubeForRegistration, Wavelengths_list, 
                                                                                                               EdgeMask=EdgeMask, #AllReflectionsMasks=AllReflectionsMasks,
                                                                                                               Blurring=True, Sigma=Sigma, StaticIndex=index,
                                                                                                               DeviationThreshold=DeviationThreshold, 
                                                                                                               AllLandmarkPoints=AllLandmarkPoints)
        temp = AllPoints['fixed_points']
        Info = f'Blurring{Sigma}_ManualRegistration_{len(temp)}points_Provided'
    else:
        print(f'Doing manual registration - Prompting user to define fixed points')
        CoregisteredHypercube, AllTransforms, CombinedMask, AllPoints = HySE.CoRegisterHypercubeAndMask_Manual(HypercubeForRegistration, Wavelengths_list, 
                                                                                                 EdgeMask=EdgeMask, #AllReflectionsMasks=AllReflectionsMasks, 
                                                                                                 Blurring=True, Sigma=Sigma, StaticIndex=index,
                                                                                                 DeviationThreshold=DeviationThreshold)
        temp = AllPoints['fixed_points']
        Info = f'Blurring{Sigma}_ManualRegistration_{len(temp)}points_UserDefined'
    
else:
    if UserDefinedROI:
        print(f'Doing SimpleITK registration, prompting user to define square ROI')
        Info = f'Blurring{Sigma}_GridSpacing{GridSpacing}_i{index}_UserDefinedROI'
    else:
        print(f'Doing SimpleITK registration on the whole image')
        Info = f'Blurring{Sigma}_GridSpacing{GridSpacing}_i{index}'
    CoregisteredHypercube, AllTransforms, CombinedMask, AllROICoordinates = HySE.CoRegisterHypercubeAndMask(HypercubeForRegistration, Wavelengths_list, 
                                                                                                            Verbose=False, Transform='BSplineTransform', 
                                                                                                            Affine=False, Blurring=True, Sigma=Sigma, 
                                                                                                            Static_Index=index, 
                                                                                                            EdgeMask=EdgeMask, #AllReflectionsMasks=AllReflectionsMasks, 
                                                                                                            GradientMagnitude=False, HistogramMatch=False, 
                                                                                                            IntensityNorm=False, Cropping=60,
                                                                                                            TwoStage=True, Metric='AdvancedMattesMutualInformation', 
                                                                                                            GridSpacing=GridSpacing, MinVal=20, 
                                                                                                            InteractiveMasks=InteractiveMasks)

```


This is what the GUI looks like:
<p align="center">
  <img src="https://github.com/user-attachments/assets/0a0aa6c3-093c-4ac7-86e1-d0f5a88fcf48" width="800"/>
</p>

Automatic error detection flags cases when the corresponding point is further away than a pre-set maximal distance. This can help prevent cases when the user makes a mistake and for example forgets a point. It is however a very blunt tool, as it will not flag errors if the points confused are close enough to each other, and it will flag errors when the movement is larger than this set threhold (can be modified when launching the GUI). The errors do not affect the code and should be ignored when the user is confident that the point are where they should be, like in this example:
<p align="center">
  <img src="https://github.com/user-attachments/assets/8cf3d3bc-9d21-40b3-bb7c-03cd382df050" width="800"/>
</p>


Then save results:
```python
NMI_Sweep_before = HySE.GetNMI(HypercubeForRegistration)
NMI_Sweep_after = HySE.GetNMI(CoregisteredHypercube)
print(f'Average NMI before : {NMI_Sweep_before[0]:.4f}, Average NMI after : {NMI_Sweep_after[0]:.4f}')

VideoSavingPath = f'{RegistrationSavingPath}Sweep{Nsweep}_Frame{Nframe}_{Info}_NMI{NMI_Sweep_after[0]:.2f}.mp4'


## Not CoRegistered
OrigVideoSavingPath = f'{RegistrationSavingPath}Sweep{Nsweep}_Frame{Nframe}_NoReg.mp4'
HySE.MakeHypercubeVideo(HypercubeForRegistration, OrigVideoSavingPath)
OrigArraySavingPath = OrigVideoSavingPath.replace('mp4', 'npz')
np.savez(f'{OrigArraySavingPath}', HypercubeForRegistration)

## Whole image
HySE.MakeHypercubeVideo(CoregisteredHypercube, VideoSavingPath, Normalise=True)
ArraySavingPath = VideoSavingPath.replace('.mp4','.npz')
np.savez(f'{ArraySavingPath}', CoregisteredHypercube)
## Masked
VideoSavingPathMasked = VideoSavingPath.replace('.mp4','_Masked.mp4')
HySE.MakeHypercubeVideo(CoregisteredHypercube, VideoSavingPathMasked, Mask=CombinedMask, Normalise=True)

# ## Masked
# VideoSavingPathMasked = VideoSavingPath.replace('.mp4','_Masked.mp4')
# HySE.MakeHypercubeVideo(CoregisteredHypercube, VideoSavingPathMasked, Mask=CombinedMask, Normalise=True)
## Mask
MaskSavingPath = VideoSavingPath.replace('.mp4','_Mask.npz')
np.savez(f'{MaskSavingPath}', CombinedMask)

## Transforms
TransformsSavingPath = VideoSavingPath.replace('.mp4','_Transforms')
HySE.SaveTransforms(AllTransforms, TransformsSavingPath)

# Wavelengths
WavelengthsSavingPath_Sorted = VideoSavingPath.replace('.mp4','__SortedWavelengths.npz') # f'{SavingPath}{Name}_{NameSub}__SortedWavelengths.npz'
WavelengthsSavingPath_Unsorted = VideoSavingPath.replace('.mp4','__UnsortedWavelengths.npz') #f'{SavingPath}{Name}_{NameSub}__UnsortedWavelengths.npz'
np.savez(f'{WavelengthsSavingPath_Sorted}', Wavelengths_list_sorted)
np.savez(f'{WavelengthsSavingPath_Unsorted}', Wavelengths_list)

if Manual:
    PointsSavingPath = VideoSavingPath.replace('.mp4','_UserDefinedPoints.npz')
    np.savez(f'{PointsSavingPath}', AllPoints, allow_pickle=True)
elif UserDefinedROI:
    ROISavingPath = VideoSavingPath.replace('.mp4','_UserDefinedROI')
    HySE.SaveTransforms(AllROICoordinates, ROISavingPath)

print(f'\n\n Saved all data')

```


#### Automatic Registration
Using SimpleITK, mutual information (metric). Does not always work very well. Can be applied after a quick manual registration.
Most implementations do first an affine transform, followed by a bspline transform. The GridSpacing paramter allows to set the unit size for the bspline transform. Too large grids do not allow to register finely enough (looks more like affine), while too small grids overfit the data/noise and introduces artefacts and distortions.

```python
# ## --- If going straight to the automatic registration: ---
# HypercubeForRegistration_Auto = HypercubeForRegistration

## OR
## --- If applying automatic registration after having done a manual pre-registration: ---

# HypercubeForRegistration_Auto = CoregisteredHypercube

## OR
## --- If Loading a previously pre-registered hypercube from path: ---

# ## Lesion2:
# TEMP = '/Users/iracicot/Library/CloudStorage/OneDrive-UniversityofCambridge/Data/HySE/Patient7_20251125/Results/'
# HypercubeForRegistration_Auto_Path = TEMP+'Lesion2_Registered/Manual_AllFeatures/Sweep3_Frame1_Blurring1_ManualRegistration_45points_Provided_NMI1.23.npz'
# HypercubeForRegistration_Auto = np.load(HypercubeForRegistration_Auto_Path)['arr_0']
# print(HypercubeForRegistration_Auto.shape)


# ## # If Averaging
# HypercubeForRegistration_Avg, _ = HySE.ComputeHypercube(DataPath, EdgePos_Data, Wavelengths_list, Buffer=Buffer, 
#                                                     Average=False, Order=False, Help=False, SaveFig=False, SaveArray=False, 
#                                                     Plot=False, ForCoRegistration=False)
# HypercubeForRegistration_Auto = HypercubeForRegistration_Avg[Nsweep,:,:,:]



Cropping = 100
Sigma = 1

index = 5

GridSpacing = 60

MinVal = 0 


#### Register:

CoregisteredHypercube_Auto, AllTransforms, CombinedMask, AllROICoordinates = HySE.CoRegisterHypercubeAndMask(HypercubeForRegistration_Auto, Wavelengths_list, 
                                                                                                        Verbose=False, Transform='BSplineTransform', 
                                                                                                        Cropping=Cropping,
                                                                                                        Affine=False, Blurring=True, 
                                                                                                        Sigma=Sigma, EdgeMask=EdgeMask,
                                                                                                        StaticIndex=index, GradientMagnitude=False, 
                                                                                                        HistogramMatch=False, IntensityNorm=False,
                                                                                                        TwoStage=True, Metric='AdvancedMattesMutualInformation', 
                                                                                                        GridSpacing=GridSpacing, MinVal=MinVal) 

Info = f'f{Nframe}_Blur{Sigma}_Grid{GridSpacing}_i{index}_RegisterAndMask'



# ####################################
# #### Compute NMI and save results:
# ####################################


NMI_Sweep_before = HySE.GetNMI(HypercubeForRegistration_Auto)
NMI_Sweep_after = HySE.GetNMI(CoregisteredHypercube_Auto)

print(f'Average NMI before : {NMI_Sweep_before[0]:.4f}, Average NMI after : {NMI_Sweep_after[0]:.4f}')



# print(f'\n\n')


VideoSavingPath = f'{RegistrationSavingPath}Sweep{Nsweep}_Frame{Nframe}_{Info}_NMI{NMI_Sweep_after[0]:.2f}.mp4'

## Not CoRegistered
OrigVideoSavingPath = f'{RegistrationSavingPath}Sweep{Nsweep}_Frame{Nframe}_NoReg.mp4'
HySE.MakeHypercubeVideo(HypercubeForRegistration_Auto, OrigVideoSavingPath)
OrigArraySavingPath = OrigVideoSavingPath.replace('mp4', 'npz')
np.savez(f'{OrigArraySavingPath}', HypercubeForRegistration_Auto)

## Whole image
HySE.MakeHypercubeVideo(CoregisteredHypercube_Auto, VideoSavingPath, Normalise=True)
ArraySavingPath = VideoSavingPath.replace('.mp4','.npz')
np.savez(f'{ArraySavingPath}', CoregisteredHypercube_Auto)
## Masked
VideoSavingPathMasked = VideoSavingPath.replace('.mp4','_Masked.mp4')
HySE.MakeHypercubeVideo(CoregisteredHypercube_Auto, VideoSavingPathMasked, Mask=CombinedMask, Normalise=True)



# ## Cropped image
# VideoSavingPathCropped = VideoSavingPath.replace('.mp4','_Cropped.mp4')
# HySE.MakeHypercubeVideo(CoregisteredHypercube_Auto[:,Cropping:-1*Cropping,Cropping:-1*Cropping], VideoSavingPathCropped, Normalise=True)


# Wavelengths
WavelengthsSavingPath_Sorted = VideoSavingPath.replace('.mp4','__SortedWavelengths.npz') # f'{SavingPath}{Name}_{NameSub}__SortedWavelengths.npz'
WavelengthsSavingPath_Unsorted = VideoSavingPath.replace('.mp4','__UnsortedWavelengths.npz') #f'{SavingPath}{Name}_{NameSub}__UnsortedWavelengths.npz'
np.savez(f'{WavelengthsSavingPath_Sorted}', Wavelengths_list_sorted)
np.savez(f'{WavelengthsSavingPath_Unsorted}', Wavelengths_list)


## Transforms
TransformsSavingPath = VideoSavingPath.replace('.mp4','_Transforms')
HySE.SaveTransforms(AllTransforms, TransformsSavingPath)

## Mask
MaskSavingPath = VideoSavingPath.replace('.mp4','_Mask.npz')
np.savez(f'{MaskSavingPath}', CombinedMask)
print(f'Saved all data')

```



## Unmixing

### Define Wavelengths and Mixing Matrices:

```python
## 3 wav split hybrid, wavelengths 1:
##                     0.   1.   2.   3.   4.   5.   6.   7
Panel2_Wavelengths = [625, 606, 584, 560, 543, 511, 486, 418]
Panel4_Wavelengths = [646, 617, 594, 576, 569, 526, 503, 478]


indices_4x4 = [1,4,5,6]
indices_3x3 = [0,2,3]
indices_6x6 = [0,1,2,3,4,5,6]
index_418 = 7

Green_Exposure = 6.5
Blue_Exposure = 7
Red_Exposure = 1.6


Green_3Wav_1 = [[4, 5, 6],
                [1, 4, 6],
                [1, 5, 6],
                [1, 4, 5],
                [4, 5, 6, 7],
                [2, 3],
                [0, 3],
                [0, 2]]

MixingMatrix_Indices = Green_3Wav_1

print(f'Computing 6x6, 4x4 and 3x3 green matrices...')
GreenMatrix_6x6_Indices, GreenMatrix_4x4_Indices, GreenMatrix_3x3_Indices = HySE.GetShortGreenMatrices_3Wav(MixingMatrix_Indices, index_418=index_418, 
                                                                                indices_4x4=indices_4x4, indices_3x3=indices_3x3)
print(f'Computing blue matrix..')
BlueMatrix_Indices = HySE.GetBlueMatrix_3Wav(MixingMatrix_Indices, index_418=index_418)

Wavelengths_list = np.array(Panel2_Wavelengths+Panel4_Wavelengths)
Wavelengths_list_sorted = np.sort(Wavelengths_list)

print(f'computing short pannels...')
Wavs_6x6_P2, Wavs_6x6_P4, Wavs_4x4_P2, Wavs_4x4_P4, Wavs_3x3_P2, Wavs_3x3_P4 = HySE.GetShortPanelWavs_3Wav(Panel2_Wavelengths, Panel4_Wavelengths, 
                                                                                                      indices_4x4, indices_3x3, index_418)

Wavs_6x6 = np.sort(np.array(Wavs_6x6_P2+Wavs_6x6_P4))

Wavs_4x4 = np.sort(np.array(Wavs_4x4_P2+Wavs_4x4_P4))
Wavs_3x3 = np.sort(np.array(Wavs_3x3_P2+Wavs_3x3_P4))

```

### Build Mixing Matrices
Note that this example builds all the mixing matrices possible for the Green channel.
```python
print(f'Green Mixing Matrix')

Title = 'Green Matrix'
GreenMatrix_7x7 = HySE.MakeMixingMatrix_Flexible(Panel2_Wavelengths, MixingMatrix_Indices, 
                                         Panel4_Wavelengths, MixingMatrix_Indices, Plot=False,
                                         SaveFig=False, Title=Title, SavingPath='')

cond_number = np.linalg.cond(GreenMatrix_7x7)
print("   Condition number:", cond_number)

#####################

print(f'Green short Mixing Matrix')
Title = 'Green Matrix 6x6'

GreenMatrix_6x6_sub_Indices = HySE.generate_local_index_matrix(GreenMatrix_6x6_Indices, indices_6x6)

GreenMatrix_6x6 = HySE.MakeMixingMatrix_Flexible(Wavs_6x6_P2, GreenMatrix_6x6_sub_Indices, 
                                         Wavs_6x6_P4, GreenMatrix_6x6_sub_Indices, Plot=False,
                                         SaveFig=False, Title=Title, SavingPath='')

cond_number = np.linalg.cond(GreenMatrix_6x6)
print("   Condition number:", cond_number)

#####################

print(f'Blue Mixing Matrix')
Title = 'Blue Matrix'
BlueMatrix = HySE.MakeMixingMatrix_Flexible(Panel2_Wavelengths, BlueMatrix_Indices, 
                                         Panel4_Wavelengths, BlueMatrix_Indices, Plot=False,
                                         SaveFig=False, Title=Title, SavingPath='')

cond_number = np.linalg.cond(BlueMatrix)
print("   Condition number:", cond_number)

#####################

MixingMatrix_Sub1 = np.array([[1,1,1,0,0,0,0],
                              [1,0,1,0,0,1,0], 
                              [1,1,0,0,0,1,0], 
                              [0,1,1,0,0,1,0], 
                              [0,0,0,1,1,0,0], 
                              [0,0,0,1,0,0,1], 
                              [0,0,0,0,1,0,1],])

MixingMatrix_Sub2 = MixingMatrix_Sub1

Wavelengths_Sub1 = np.array([486,511,543,560,584,606,625])
Wavelengths_Sub2 = np.array([503,526,569,576,594,617,646])


Sub1_indices = [6,5,4,3,2,1,0]
Sub2_indices = [14,13,12,11,10,9,8]

Sub1_indices_frames = [0,1,2,3,5,6,7]
Sub2_indices_frames = [8,9,10,11,13,14,15]

#####################

MixingMatrix_SubSub1A = np.array([[1,1,1,0],
                                  [1,0,1,1],
                                  [1,1,0,1],
                                  [0,1,1,1]])

MixingMatrix_SubSub1B = np.array([[1,1,0],
                                  [1,0,1],
                                  [0,1,1]])

MixingMatrix_SubSub2A = MixingMatrix_SubSub1A
MixingMatrix_SubSub2B = MixingMatrix_SubSub1B

Wavelengths_SubSub1A = np.array([486,511,543,606])
Wavelengths_SubSub1B = np.array([560,584,625])
Wavelengths_SubSub2A = np.array([503,526,569,617])
Wavelengths_SubSub2B = np.array([576,594,646])

SubSub1A_indices_frames = [0,1,2,3]
SubSub1B_indices_frames = [5,6,7]
SubSub2A_indices_frames = [8,9,10,11]
SubSub2B_indices_frames = [13,14,15]

```
<p align="center">
  <img src="https://github.com/user-attachments/assets/49fd5867-b39b-40b0-8cfc-38f39294e873" width="500"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/650e1403-c03a-44b0-9424-341d801b709a" width="850"/>
</p>



### Locate Data

```python
GeneralPath = '/GENERAL_PATH/'

DataPath_General = GeneralPath+'Videos/'
DarkPath = DataPath_General+'2025-11-25_08-25-28_Dark.mp4'


## Lesion
DataPath = DataPath_General+'Lesion.mp4'
Name = 'Lesion'

TransformsPath = GeneralPath+'Results/Lesion1_Registered/'
Nsweep = 3
Nframe = 1
RegisteredName = f'Sweep{Nsweep}f{Nframe}_Manual'

```

### Import Transforms, Dark and Trace

```python
Transforms = HySE.LoadTransforms(TransformsPath)
print(len(Transforms))
LongDark = HySE.GetDark_WholeVideo(DarkPath) #, CropImDimensions=CropImDimensions
EdgePos_Data = HySE.FindHypercube_RGB(DataPath, Automatic=True, PlateauSize=9, DarkMin=12, MaxPlateauSize=20, SaveFig=False)
```


### Extract Frames and Subtract Dark
```python
Buffer = 3
Frames, RGB_Dark = HySE.ComputeHypercube_RGB(DataPath, EdgePos_Data, Buffer=Buffer, BlueShift=1, SaveArray=False)

## Data

(Nsweeps, Nwav, Nframes, Y, X, _) = Frames.shape

## BGR 
Frames_Blue = Frames[:,:,:,:,:,0]
Frames_Green = Frames[:,:,:,:,:,1]
Frames_Red = Frames[:,:,:,:,:,2]

# NormaliseMixedHypercube
Frames_BlueD, MaskB = HySE.NormaliseMixedHypercube(Frames_Blue, Dark=LongDark, SaveFigure=False, Plot=False)
Frames_GreenD, MaskG = HySE.NormaliseMixedHypercube(Frames_Green, Dark=LongDark, SaveFigure=False, Plot=False)
Frames_RedD, MaskR = HySE.NormaliseMixedHypercube(Frames_Red, Dark=LongDark, SaveFigure=False, Plot=False)

print(Frames_GreenD.shape)

```

### Apply Transforms to Data
The transforms are determined using the red frames, which are all obtained using the same illumination.
The data frames are in the green and blue channels. It is assumed that the movement between RGB frames is minimal, so that transforms from the red frames is applicable to equivalement GB frames.
```python
print(f'Taking a subset of normalised frames for sweep {Nsweep}, frame {Nframe}')
Frames_GreenD_Sub = Frames_GreenD[Nsweep, :, Nframe, :, :]
Frames_BlueD_Sub = Frames_BlueD[Nsweep, :, Nframe, :, :]

RegFrames_GreenD_Sub = HySE.ApplyTransform(Frames_GreenD_Sub, Transforms)
RegFrames_BlueD_Sub = HySE.ApplyTransform(Frames_BlueD_Sub, Transforms)
```

### Select Approprite frames to unmix in smaller chunks
Only possible for the green channel
```python
## Green
FramesG_6x6 = HySE.omit_frames(RegFrames_GreenD_Sub, indices_6x6)

FramesG_Sub1 = RegFrames_GreenD_Sub[Sub1_indices_frames]
FramesG_Sub2 = RegFrames_GreenD_Sub[Sub2_indices_frames]

FramesG_SubSub1A = RegFrames_GreenD_Sub[SubSub1A_indices_frames]
FramesG_SubSub1B = RegFrames_GreenD_Sub[SubSub1B_indices_frames]
FramesG_SubSub2A = RegFrames_GreenD_Sub[SubSub2A_indices_frames]
FramesG_SubSub2B = RegFrames_GreenD_Sub[SubSub2B_indices_frames]
```

### Unmix
Note that these steps do all the possible ways the data can be unmixed. It can be shorten significantly by only doing one kind of unmixing (one for the blue channel, one for the green channel)
There are 3 functions in this package that can be used for umxing:
- UnmixData(): Basic matrix inversion using least squares
- UnmixDataNNLS(): Matrix inversition using non-negative least squares (NNLS), with noise filtering and rejection of pixels with negative or invalid intensity
- UnmixDataSmoothNNLS(): Matrix inversion using NNLS with with noise filtering and pixel rejection, imposing smoothness (regulation set weighted by lambda)
- UnmixDataSmoothNNLSPrior(): Matrix inversino using NNLS with two regulators, smoothnes (weighted by lambda_smooth) and a known spectral prior (weighted by lambda_prior)

The UnmixDataSmoothNNLS() function generally leads to the best results, with low lambda (~0.1)
```python
lambda_value = 0.1


## Blue Matrix
t0 = time.time()
Unmixed_Blue = HySE.UnmixDataSmoothNNLS(RegFrames_BlueD_Sub, BlueMatrix, lambda_smooth=lambda_value)
t1 = time.time()
print(f'\nFull Blue - Smooth NNLS unmixing took {t1-t0:.0f}s')



## Green Matrix, all frames:
t0 = time.time()
Unmixed_Green_7x7 = HySE.UnmixDataSmoothNNLS(RegFrames_GreenD_Sub, GreenMatrix_7x7, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Full Green - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, 6x6:
t0 = time.time()
Unmixed_Green_6x6 = HySE.UnmixDataSmoothNNLS(FramesG_6x6, GreenMatrix_6x6, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green 6x6 - Smooth NNLS unmixing took {t1-t0:.0f}s')


## Green Matrix, Sub1:
t0 = time.time()
Unmixed_Green_Sub1 = HySE.UnmixDataSmoothNNLS(FramesG_Sub1, MixingMatrix_Sub1, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green Sub1 - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, Sub2:
t0 = time.time()
Unmixed_Green_Sub2 = HySE.UnmixDataSmoothNNLS(FramesG_Sub2, MixingMatrix_Sub2, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green Sub2 - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, SubSub1A:
t0 = time.time()
Unmixed_Green_SubSub1A = HySE.UnmixDataSmoothNNLS(FramesG_SubSub1A, MixingMatrix_SubSub1A, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green SubSub1A - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, SubSub1B:
t0 = time.time()
Unmixed_Green_SubSub1B = HySE.UnmixDataSmoothNNLS(FramesG_SubSub1B, MixingMatrix_SubSub1B, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green SubSub1B - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, SubSub2A:
t0 = time.time()
Unmixed_Green_SubSub2A = HySE.UnmixDataSmoothNNLS(FramesG_SubSub2A, MixingMatrix_SubSub2A, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green SubSub2A - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, SubSub2B:
t0 = time.time()
Unmixed_Green_SubSub2B = HySE.UnmixDataSmoothNNLS(FramesG_SubSub2B, MixingMatrix_SubSub2B, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green SubSub2B - Smooth NNLS unmixing took {t1-t0:.0f}s')

# print(f'All the unmixing took {t1-t0:.0f}s')
```

### Recombine frames that were unmixed separatly (Green)
```python
## in two parts:
Unmixed_GreenSub12_Combined, Wavs_Sub12_Combined = HySE.combine_hypercubes(Unmixed_Green_Sub1, Unmixed_Green_Sub2, 
                                                                           Wavelengths_Sub1, Wavelengths_Sub2)



## in four parts:
Unmixed_GreenSubSub1_Combined, Wavs_SubSub1_Combined = HySE.combine_hypercubes(Unmixed_Green_SubSub1A, Unmixed_Green_SubSub1B, 
                                                                               Wavelengths_SubSub1A, Wavelengths_SubSub1B)
Unmixed_GreenSubSub2_Combined, Wavs_SubSub2_Combined = HySE.combine_hypercubes(Unmixed_Green_SubSub2A, Unmixed_Green_SubSub2B, 
                                                                               Wavelengths_SubSub2A, Wavelengths_SubSub2B)

Unmixed_GreenSubSubAll_Combined, Wavs_SubSubAll_Combined = HySE.combine_hypercubes(Unmixed_GreenSubSub1_Combined, Unmixed_GreenSubSub2_Combined, 
                                                                                   Wavs_SubSub1_Combined, Wavs_SubSub2_Combined)
```

### Save Results
```python
## Wavelengths (for GUI)
np.savez(SavingPath+'wavelengths_sub.npz', Wavs_Sub12_Combined)
np.savez(SavingPath+'wavelengths.npz', Wavelengths_list_sorted)

## Blue
np.savez(SavingPath+f'{Name}_{RegisteredName}_Umixed_Blue.npz', Unmixed_Blue)

## Green
np.savez(SavingPath+f'{Name}_{RegisteredName}_Umixed_Green_7x7.npz', Unmixed_Green_7x7)
np.savez(SavingPath+'Umixed_Green_6x6.npz', Unmixed_Green_6x6)

np.savez(SavingPath+f'{Name}_{RegisteredName}_Umixed_Green_Split2.npz', Unmixed_GreenSub12_Combined)
np.savez(SavingPath+f'{Name}_{RegisteredName}_Umixed_Green_Split4.npz', Unmixed_GreenSubSubAll_Combined)

HySE.MakeHypercubeVideo(Unmixed_Blue, SavingPath+f'{Name}_{RegisteredName}_Umixed_Blue.mp4', Normalise=True)
HySE.MakeHypercubeVideo(RegFrames_BlueD_Sub, SavingPath+f'{Name}_{RegisteredName}_RegMixed_Blue.mp4', Normalise=True)
HySE.MakeHypercubeVideo(Frames_BlueD_Sub, SavingPath+f'{Name}_{RegisteredName}_Raw_Blue.mp4', Normalise=True)

HySE.MakeHypercubeVideo(Unmixed_GreenSub12_Combined, SavingPath+f'{Name}_{RegisteredName}_Umixed_Green_Split2.mp4', Normalise=True)
HySE.MakeHypercubeVideo(RegFrames_GreenD_Sub, SavingPath+f'{Name}_{RegisteredName}_RegMixed_Green.mp4', Normalise=True)
HySE.MakeHypercubeVideo(Frames_GreenD_Sub, SavingPath+f'{Name}_{RegisteredName}_Raw_Green.mp4', Normalise=True)
```

### For Visualisation:
Save Hypercube Figures
```python
# ## Blue
# Unmixed_Blue

# ## Green
# Unmixed_Green_7x7
# Unmixed_Green_6x6
# Unmixed_GreenSub12_Combined
# Unmixed_GreenSubSubAll_Combined

# ## Wavelenghts
# Wavs_Sub12_Combined
# Wavelengths_list_sorted

SavingPathWithName = f'{SavingPath}{Name}_{RegisteredName}_Hypercube_Green_Split4.png'
HySE.PlotHypercube(Unmixed_GreenSubSubAll_Combined[:,90:900,45:995], Wavelengths=Wavs_Sub12_Combined, SameScale=False,# vmax=30,
                   SavePlot=True, ShowPlot=False, SavingPathWithName=SavingPathWithName)

SavingPathWithName = f'{SavingPath}{Name}_{RegisteredName}_Hypercube_Green_Split2.png'
HySE.PlotHypercube(Unmixed_GreenSub12_Combined[:,90:900,45:995], Wavelengths=Wavs_Sub12_Combined, SameScale=False,# vmax=30,
                   SavePlot=True, ShowPlot=False, SavingPathWithName=SavingPathWithName)

SavingPathWithName = f'{SavingPath}{Name}_{RegisteredName}_Hypercube_Green_6x6.png'
HySE.PlotHypercube(Unmixed_Green_6x6[:,90:900,45:995], Wavelengths=Wavs_Sub12_Combined, SameScale=False,# vmax=30,
                   SavePlot=True, ShowPlot=False, SavingPathWithName=SavingPathWithName)

SavingPathWithName = f'{SavingPath}{Name}_{RegisteredName}_Hypercube_Green_7x7.png'
HySE.PlotHypercube(Unmixed_Green_7x7[:,90:900,45:995], Wavelengths=Wavelengths_list_sorted, SameScale=False,# vmax=30,
                   SavePlot=True, ShowPlot=False, SavingPathWithName=SavingPathWithName)

SavingPathWithName = f'{SavingPath}{Name}_{RegisteredName}_Hypercube_Blue.png'
HySE.PlotHypercube(Unmixed_Blue[:,90:900,45:995], Wavelengths=Wavelengths_list_sorted, SameScale=False,# vmax=30,
                   SavePlot=True, ShowPlot=False, SavingPathWithName=SavingPathWithName)
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/5a08ee78-a088-4881-aba4-c79da1dfaa03" width="550"/>
</p>


## Visualisation

### BiopsyPicker
This GUI allows to draw ROIs where the user has determined the biopsies were taken.

As with the other PyQT GUI, the correct matplotlib settings must be set:
``` python
%matplotlib qt
```

The registerd (but ideally not yet unmixed) hypercube must be loaded:
``` python
# Lesion
GeneralPath = '/PATH_TO_DATA/'
SavingPath = GeneralPath

CoregisteredHypercube_Path = GeneralPath+'RegisteredLesion.npz'
Name = 'Sweep3_Frame1'
Wavelengths_Path = GeneralPath+'wavelengths.npz'

CoregisteredHypercube = np.load(CoregisteredHypercube_Path)['arr_0']
Wavelengths_list = np.load(Wavelengths_Path)['arr_0']
```
Before being fed to the function.
Note that previously defined points can be loaded as arguments to this function for modifications.
``` python
ROI_coordinates, ROI_AvgSpectra, ROI_AllSpectra, preview_img = HySE.GetBiopsyLocations(CoregisteredHypercube)
```
Pressing "r" allowd to draw one ROI by clicking on the image. The ROI is set by closing the loop (clicking back on the first point). Double clicking on any ROI allows to modify it. Points can be modified by draging them somewhere else, removed by double-clicking on them, or added by double-clicking on the image. Pressing "e" exist the ROI modifying mode. 

Once the window is closed, the ROI parameters are output the following way:
ROI_coordinates: List of lenght N_rois. Each element is a 2D array of size [Npoints, 2] for the x and y coordinates
ROI_AvgSpectra: List of lenght N_rois. Each element is the average spectral in the ROI (length Nwavelengths)
ROI_AllSpectra: List of lenght N_rois. Each element is a 2D array of size [Npixels, Nwavelengths] containing the spectral for each pixel in the ROI
preview_img: image of a frame with the ROIs overlapped. To be saved as a png image for quick reference. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/978406ad-5385-4458-8e22-ab2f0f2e7db8" width="550"/>
</p>

Save the results for future references:
```python
plt.close()
fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.imshow(preview_img)
ax.axis('off')
ax.set_title('Identified ROIs', fontsize=14)
plt.tight_layout()
plt.savefig(f'{SavingPath}{Name}_ROIs_Preview.png')
plt.show()


ROIs = [
    dict(
        Coordinates=c,
        AvgSpectra=a,
        AllSpectra=s
    )
    for c, a, s in zip(ROI_coordinates, ROI_AvgSpectra, ROI_AllSpectra)
]

ROI_SavingPath = f'{SavingPath}{Name}_ROIs.npz'
np.savez_compressed(ROI_SavingPath, ROIs=np.array(ROIs, dtype=object))
```


### Registration Visualisation
Those functions allow to visualise the registration.

Load the unregistered and registered hypercubes:
```python
# Lesion
GeneralPath = '/GENERAL_PATH_LESION/'
SavingPath = GeneralPath

HypercubeForRegistration_Path = GeneralPath+'Sweep3_Frame1_NoReg.npz'
CoregisteredHypercube_Path = GeneralPath+'Sweep3_Frame1_Blurring1_ManualRegistration_45points_Provided_NMI1.23.npz'
Name = 'Sweep3_Frame1'

RegisteredMask_Path = GeneralPath+'Sweep3_Frame1_Blurring1_ManualRegistration_45points_Provided_NMI1.23_Mask.npz'
Wavelengths_Path = GeneralPath+'wavelengths.npz'

HypercubeForRegistration = np.load(HypercubeForRegistration_Path)['arr_0']
CoregisteredHypercube = np.load(CoregisteredHypercube_Path)['arr_0']
RegisteredMask = np.load(RegisteredMask_Path)['arr_0']
Wavelengths_list = np.load(Wavelengths_Path)['arr_0']

```

Specular reflections dominate the sobel (edge detection). They must be filtered out
```python
Unregistered = HypercubeForRegistration
Registered = CoregisteredHypercube


Unregistered_ = np.clip(Unregistered, 0, None)

## Sweep 13:
##.        0.   1.   2.   3.   4.   5   6.   7.     8.   9.   10.  11.  12.  13.  14.  15. 
k_list = [4.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0,   2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0]

HypercubeForRegistration_Masked, AllReflectionsMasks = HySE.RemoveSpecularReflections(Unregistered_, k=k_list, FillValue=0, Buffer=4, MaxSize=2800)
print(AllReflectionsMasks.shape)

fs=11
## Make sure reflections are masked appropriately
# Indices = [0,5,14]
Indices = [i for i in range(0,len(Wavelengths_list))]

plt.close()
fig, ax = plt.subplots(nrows=len(Indices), ncols=3, figsize=(10,2.5*len(Indices)))
for i in range(0,len(Indices)):
    u = Indices[i]
    m, M = np.nanmin(Unregistered_[u,:,:]), np.nanmax(Unregistered_[u,:,:])
    ax[i,0].imshow(Unregistered_[u,:,:], vmin=m, vmax=M, cmap='gray')
    ax[i,0].set_title(f'Hypercube u={u}', fontsize=fs)
    ax[i,1].imshow(HypercubeForRegistration_Masked[u,:,:], vmin=m, vmax=M, cmap='gray')
    ax[i,1].set_title(f'Masked hypercube u={u} - k={k_list[u]}', fontsize=fs)
    ax[i,2].imshow(AllReflectionsMasks[u,:,:])
    ax[i,2].set_title(f'Mask u={u} - Sum = {np.sum(AllReflectionsMasks[u,:,:])}', fontsize=fs)
    for j in range(0,3):
        ax[i,j].set_axis_off()
plt.tight_layout()
plt.show()
```

This function plots just the edges
```python

ReflectionsMask = AllReflectionsMasks
RegisteredMask = RegisteredMask

## Lesion
UnregCoords = [200,-45, 45,-45] ## ys,ye, xs,xe
RegCoords = [200,-45, 45,-45]


method = 'sobel'                   # sobel', 'canny', or 'raw_thresholded'.
sobel_ksize = 3
canny_sigma = 2.0                  # Sigma for Canny edge detection
cap_intensity = 0.2                # Clips high-intensity features (0.0 to 1.0)
raw_threshold = 0.4                # Threshold for 'raw_thresholded' -> Set everything below to 0
nframe = 1                         # Context frame index
mask_dilation_kernel = 10          # Size of dilation for reflection mask.
denoise_sigma = 0                  # Pre-filtering strength.
denoise_method = 'gaussian'          # 'gaussian' or 'median'.
mask_inpaint_method = 'inpaint'    # 'zero_fill' or 'inpaint'.
display_power_gamma = 2.0          # Power law (val^gamma) to darken background noise.
min_gradient_threshold = 0.20      # Hard cutoff (0.0 to 1.0). Signals below this are set to 0.

plt.close()
plt.close()

fig_unregistered, im_unregistered =  HySE.visualize_hypercube_movement(Unregistered[:,UnregCoords[0]:UnregCoords[1],UnregCoords[2]:UnregCoords[3]], 
                                    method=method,
                                    sobel_ksize=sobel_ksize, 
                                    canny_sigma=canny_sigma,
                                    cap_intensity=cap_intensity, 
                                    raw_threshold=raw_threshold, 
                                    reflection_mask=ReflectionsMask[:,UnregCoords[0]:UnregCoords[1],UnregCoords[2]:UnregCoords[3]], 
                                    nframe=nframe, 
                                    mask_dilation_kernel=mask_dilation_kernel, 
                                    denoise_sigma=denoise_sigma, 
                                    denoise_method=denoise_method,
                                    mask_inpaint_method=mask_inpaint_method, 
                                    display_power_gamma=display_power_gamma,
                                    min_gradient_threshold=min_gradient_threshold)



fig_registered, im_registered =  HySE.visualize_hypercube_movement(Registered[:,RegCoords[0]:RegCoords[1],RegCoords[2]:RegCoords[3]], 
                                    method=method,
                                    sobel_ksize=sobel_ksize,
                                    canny_sigma=canny_sigma, 
                                    cap_intensity=cap_intensity, 
                                    raw_threshold=raw_threshold, 
                                    reflection_mask=RegisteredMask[RegCoords[0]:RegCoords[1],RegCoords[2]:RegCoords[3]], 
                                    nframe=nframe, 
                                    mask_dilation_kernel=4, 
                                    denoise_sigma=denoise_sigma, 
                                    denoise_method=denoise_method,
                                    mask_inpaint_method=mask_inpaint_method, 
                                    display_power_gamma=display_power_gamma,
                                    min_gradient_threshold=min_gradient_threshold)

```
```python
plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12,4))

ax[0].imshow(fig_unregistered)
ax[0].axis('off')
ax[0].set_title('Unregistered', fontsize=14)

ax[1].imshow(fig_registered)
ax[1].axis('off')
ax[1].set_title('Registered', fontsize=14)

plt.tight_layout()
plt.savefig(f'{SavingPath}{Name}_RegistrationCommparison.png', transparent=True)
plt.savefig(f'{SavingPath}{Name}_RegistrationCommparison.svg', transparent=True)
plt.show()
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/62a9f037-b6ed-4032-ad89-f6aff8c826b7" width="750"/>
</p>

While this function overlays the edges on top of a greyscale image of the lesion:
```python

method = 'sobel'                   # sobel', 'canny', or 'raw_thresholded'.
sobel_ksize = 15
canny_sigma = 2.0                  # Sigma for Canny edge detection
cap_intensity = 0.005                # Clips high-intensity features (0.0 to 1.0)
raw_threshold = 0.4                # Threshold for 'raw_thresholded' -> Set everything below to 0
nframe = 1                         # Context frame index
mask_dilation_kernel = 10          # Size of dilation for reflection mask.
denoise_sigma = 0                  # Pre-filtering strength.
denoise_method = 'gaussian'          # 'gaussian' or 'median'.
mask_inpaint_method = 'inpaint'    # 'zero_fill' or 'inpaint'.
min_gradient_threshold = 0.20      # Hard cutoff (0.0 to 1.0). Signals below this are set to 0.

opacity_gain = 1.2

plt.close()
plt.close()

overlay_unregistered =  HySE.visualize_edge_density_overlay(Unregistered[:,UnregCoords[0]:UnregCoords[1],UnregCoords[2]:UnregCoords[3]], 
                                                       sobel_ksize=sobel_ksize, 
                                                       cap_intensity=cap_intensity, 
                                                       raw_threshold=raw_threshold,
                                                       reflection_mask=ReflectionsMask[:,UnregCoords[0]:UnregCoords[1],UnregCoords[2]:UnregCoords[3]],
                                                       mask_dilation_kernel=mask_dilation_kernel,
                                                       denoise_sigma=denoise_sigma,
                                                       denoise_method=denoise_method,
                                                       mask_inpaint_method=mask_inpaint_method,
                                                       min_gradient_threshold=min_gradient_threshold,
                                                       opacity_gain=opacity_gain)

overlay_registered =  HySE.visualize_edge_density_overlay(Registered[:,RegCoords[0]:RegCoords[1],RegCoords[2]:RegCoords[3]], 
                                                       sobel_ksize=sobel_ksize, 
                                                       cap_intensity=cap_intensity, 
                                                       raw_threshold=raw_threshold,
                                                       reflection_mask=RegisteredMask[RegCoords[0]:RegCoords[1],RegCoords[2]:RegCoords[3]], 
                                                       mask_dilation_kernel=mask_dilation_kernel,
                                                       denoise_sigma=denoise_sigma,
                                                       denoise_method=denoise_method,
                                                       mask_inpaint_method=mask_inpaint_method,
                                                       min_gradient_threshold=min_gradient_threshold,
                                                       opacity_gain=opacity_gain)
```
```python
plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12,4))

ax[0].imshow(overlay_unregistered)
ax[0].axis('off')
ax[0].set_title('Unregistered', fontsize=14)

ax[1].imshow(overlay_registered)
ax[1].axis('off')
ax[1].set_title('Registered', fontsize=14)

plt.tight_layout()
plt.savefig(f'{SavingPath}{Name}_RegistrationOverlay.png', transparent=True)
plt.savefig(f'{SavingPath}{Name}_RegistrationOverlay.svg', transparent=True)
plt.show()
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/96662f5e-df99-4f84-97c8-0266aba6cb56" width="750"/>
</p>



















```

## Optional Functions

### Macbeth Colour Chart analysis

When imaging a Macbeth colour chart, it is helpful to look at the spectra from each patch.
Certain functions are designed to make this task easier.

```python

## Location of ground truth files

MacbethPath = 'MacBeth/Micro_Nano_CheckerTargetData.xls'

Macbeth_sRGBPath = 'MacBeth/Macbeth_sRGB.xlsx'
Macbeth_AdobePath = 'MacBeth/Macbeth_Adobe.xlsx'

MacBethSpectraData = np.array(pd.read_excel(MacbethPath, sheet_name='Spectra'))
MacBethSpectraColour = np.array(pd.read_excel(MacbethPath, sheet_name='Color_simple'))

MacBeth_RGB = np.array(pd.read_excel(Macbeth_AdobePath))
## Adobe RGB
MacBethSpectraRGB = np.array([MacBethSpectraColour[:,0], MacBethSpectraColour[:,7], MacBethSpectraColour[:,8], MacBethSpectraColour[:,9]])


## Crop around macbeth chart

xs, xe = 170,927
ys, ye = 310,944

macbeth = Unmixed_Hypercube_MacbethHySE_avg_ND[-1,ys:ye,xs:xe]

plt.close()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5))
ax.imshow(macbeth, cmap='gray')
plt.tight_layout()
plt.show()


## Identify where each patch is located

## This step requires referncing the plot previously generated, and usually requires a few iterations for each new dataset
## Make sure that the ROI squares cover the right patches and only those, by looking at the generated plot.
## The larger the ROI, the more pixels can be averaged to reduce noise
Patch1_pos = [558, 660]
## Size of a single macbeth patch, in pixel
Patch_size_x = 116
Patch_size_y = 115
Sample_size = 50
Image_angle = 0
 
Positions = HySE.GetPatchPos(Patch1_pos, Patch_size_x, Patch_size_y, Image_angle)
HySE.PlotPatchesDetection(macbeth, Positions, Sample_size)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/2fc5df12-4370-46ab-9119-1664b781fe36" width="400"/>
</p>

```python

## Compute the spectra for each patch

CropCoordinates = [xs, xe, ys, ye]

PatchesSpectra_Green7x7 = HySE.GetPatchesSpectrum(Unmixed_Green_7x7, Sample_size, Positions, CropCoordinates)
PatchesSpectra_Green6x6 = HySE.GetPatchesSpectrum(Unmixed_Green_6x6, Sample_size, Positions, CropCoordinates)
PatchesSpectra_Green4x4 = HySE.GetPatchesSpectrum(Unmixed_Green_4x4, Sample_size, Positions, CropCoordinates)
PatchesSpectra_Green3x3 = HySE.GetPatchesSpectrum(Unmixed_Green_3x3, Sample_size, Positions, CropCoordinates)

PatchesSpectra_Blue = HySE.GetPatchesSpectrum(Unmixed_Blue, Sample_size, Positions, CropCoordinates)
PatchesSpectra_GreenSub_Combined = HySE.GetPatchesSpectrum(Unmixed_GreenSub12_Combined, Sample_size, Positions, CropCoordinates)
PatchesSpectra_GreenSubSub_Combined = HySE.GetPatchesSpectrum(Unmixed_GreenSubSub_Combined, Sample_size, Positions, CropCoordinates)

## Plot

PatchesToPlot = [PatchesSpectra_Blue,
                 PatchesSpectra_Green7x7, 
                 PatchesSpectra_Green6x6,
                 PatchesSpectra_GreenSub_Combined, 
                 PatchesSpectra_GreenSubSub_Combined]

WavelengthsAll = [Wavelengths_list_sorted, 
                  Wavelengths_list_sorted,
                  Wavs_6x6,  
                  Wavs_6x6, 
                  Wavs_6x6]

Labels = ['Blue', 
          'Green full',
          'Green 6x6',  
          'Green sub combined', 
          'Green sub sub combined']

Colours = ['royalblue', 
           'forestgreen', 
           'limegreen', 
           'deeppink', 
           'cyan']


metric = 'SAM'

Name_ToSave = f'{SavingPath}{Name}'
HySE.PlotPatchesSpectra(PatchesToPlot, WavelengthsAll, MacBethSpectraData, MacBeth_RGB, Name, PlotLabels=Labels, SavingPath=Name_ToSave, XScale=[400,670],
                      Metric=metric, ChosenMethod=2, SaveFig=False, Colours=Colours)

```

<p align="center">
  <img src="https://github.com/user-attachments/assets/5fa369a3-d25c-49de-9eb7-1788d9319b4f" width="800"/>
</p>



<details>
<summary><b>Older Registration Documentation</b></summary>

## Co-Registration
The co-registration must be done before the unmixing. But since it requires SimpleITK and is only necessary for in vivo data (as opposed to testing data), I will for now put the relevant documentation here.
TO DO: Update description to incroporate registration at the right spot.

Co-registration is done with SimpleITK, which is an open-source interface trying to make the tooks of the the Insight Segmentation and Registration Toolkit (ITK) easier to use. It also uses functionalities from elastix, but all those are handled by SimpleITK directly.

The following documation is preliminary and indicates how to reproduce the best co-registration results obtained from HySE in vivo data, and gives a rough overview of all the options now available to play with.

Assuming that a video has been loaded, the trace computed and EdgePos computed with the HySE.FindHypercube() function, we may then start co-registering individual sweeps.

### Get raw hypercube for co-registration
This can be done either by averaging a given set of frames, or selecting a signle (good) frame. It is assumed that movement between individual RGB cycles is minimal (valid most of the time unless chromatic effects can be seen on the endoscopy monitor, which is very rare). Note that RGB frames have different exposure times, so there is typically one that has better SNR.

The sweep of interest must first be identified. Make sure that there are good frames (bubble-free, with good visibility) for all 16 wavelengths combinatino, and that movement is not too large that the same features can be identified in the whole sweep.

```python
Nsweep = 5

## ## If taking single frames
Nframe = 1
## (or Nframe = [1,2] if we want to co-register more frames - Note that this will result of a 'hypercube' or size (len(Nframe)*16,Y,X) )
HypercubeForRegistration = HySE.GetHypercubeForRegistration(Nsweep, Nframe, Lesion1_Path, EdgePos_Lesion1, Wavelengths_list, Buffer=12)


## ## If Averaging
HypercubeForRegistration_Avg, _ = HySE.ComputeHypercube(Lesion1_Path, EdgePos_Lesion1, Wavelengths_list, Buffer=Buffer, 
                                                    Average=False, Order=False, Help=False, SaveFig=False, SaveArray=False, 
                                                    Plot=False, ForCoRegistration=False)
HypercubeForRegistration = HypercubeForRegistration_Avg[Nsweep,:,:,:]
```

### Get Mask
#### Edges
Co-registration performs better when the sharp black edges from the endoscopy monitor display are masked

```python
## Use the data iself, or any normalisation, to estimate the sharp edges mask (anything works)
_, Mask = HySE.NormaliseMixedHypercube(Hypercube_Lesion1_all[2,:,:,:], Dark=LongDark, Wavelengths_list=Wavelengths_list, 
                                       SaveFigure=False, SavingPath=SavingPath+Name, vmax=160, Plot=False)

## Get the best mask to remove the edges of the image
EdgeMask = HySE.GetBestEdgeMask(Mask)
```

#### Specular Reflections
For some lesions, wet mucosa will lead to specular reflections, which can corrupt the registration. Those reflections have to be identified and masked for each frame.

```python
## Work on a single sweep that has been confirmed to have clear frames for the whole sweep, with minimal movement. Select a frame to work with (do not average images). If possible, use the (red) reference channel frames (which use all the same wavelengths and should be bright enough)
Nsweep = 13
Nframe = 1
## If necessary, clip the data to ensure no negative values (which might happen after dark subtraction)
HypercubeForRegistration = np.clip(Frames_GreenD[Nsweep, :, Nframe, :,:], 0, None)
```

Use the RemoveSpecularReflections() function to find masks to remove the specular reflections. If required (it probably will), you can upload a list of k values (lenght = # of frames) for to fine-tune the cutoff for each frame. The function accepts two methods: threshold (determined by k or cutoff, if specified. Lower k=most masking) and neighbourhood value (if NeighborhoodSize is specified). Use HySE.help('Masking.RemoveSpecularReflections') for details.

```python
## Find the right parameters to mask all specular reflections
## N
## Sweep 13:
##.        0.   1.   2.   3.   4.   5   6.   7.     8.   9.   10.  11.  12.  13.  14.  15. 
k_list = [3.1, 4.0, 3.1, 3.8, 3.8, 3.5, 4.0, 3.8,   3.7, 3.7, 3.6, 3.7, 3.7, 3.7, 4.0, 3.8]

HypercubeForRegistration_Masked, AllReflectionsMasks = HySE.RemoveSpecularReflections(HypercubeForRegistration, k=k_list, FillValue=0, Buffer=5) #FillValue=0, NeighborhoodSize=20, , Cutoff=200
print(AllReflectionsMasks.shape)
```
The fine-tuning of parameters will require some trial and error. Use the following code to visualise results:
```python
fs=11
## Make sure reflections are masked appropriately
Indices = [i for i in range(0,len(Wavelengths_list))]

fig, ax = plt.subplots(nrows=len(Indices), ncols=3, figsize=(10,2.5*len(Indices)))
# plt.close()
for i in range(0,len(Indices)):
    u = Indices[i]
    m, M = np.nanmin(HypercubeForRegistration[u,:,:]), np.nanmax(HypercubeForRegistration[u,:,:])
    ax[i,0].imshow(HypercubeForRegistration[u,:,:], vmin=m, vmax=M, cmap='gray')
    ax[i,0].set_title(f'Hypercube u={u}', fontsize=fs)
    ax[i,1].imshow(HypercubeForRegistration_Masked[u,:,:], vmin=m, vmax=M, cmap='gray')
    ax[i,1].set_title(f'Masked hypercube u={u} - k={k_list[u]}', fontsize=fs)
    ax[i,2].imshow(AllReflectionsMasks[u,:,:])
    ax[i,2].set_title(f'Mask u={u} - Sum = {np.sum(AllReflectionsMasks[u,:,:])}', fontsize=fs)
    for j in range(0,3):
        ax[i,j].set_axis_off()
plt.tight_layout()
plt.show()
```
An example of a subset of the figure generated can be seen here:
<p align="center">
  <img src="https://github.com/user-attachments/assets/03f16e61-9c8e-4af1-9c39-feaffbbb158d" width="800"/>
</p>

It's best to save those masks for furture registration. Registration codes will accept the EdgeMask and the AllReflectionsMasks.
```python
ArraySavingPath = f'{SavingPath}{Name}_{NameSub}_RawFrames/Sweep{Nsweep}_HypercubeForRegistration.npz'
np.savez(f'{ArraySavingPath}', HypercubeForRegistration)
ArraySavingPath = f'{SavingPath}{Name}_{NameSub}_RawFrames/Sweep{Nsweep}_AllReflectionsMasks.npz'
np.savez(f'{ArraySavingPath}', AllReflectionsMasks)
ArraySavingPath = f'{SavingPath}{Name}_{NameSub}_RawFrames/Sweep{Nsweep}_EdgeMask.npz'
np.savez(f'{ArraySavingPath}', EdgeMask)
```


### Perform Co-Registration

Then the co-registration. You will need to set:

- GridSpacing: This is the size of the grid used in the BSpline coregistration. Small grid spacings can co-registrate details better but tend to overfit and create distortions and artefacts. Large grid size preserve the image better but can lead to poorer registration. Some of the pattern artefact created by smaller grid spacings can be averaged out with appropriate blurring. GradSpacing of 20-50 has lead to the best results (100+ tends to not co-register everything)
- Sigma: This is the width of the Gaussian blurring function (in pixels) used to blur the image. Blurring is done twice, once on the images fed to get the coregistration transform, in order to smooth out noise that could mess with the registration, and once after registration to remove any artefact potentially introduced by the registration. Sigma of 1 has given good results.
- TwoStage: Set to true to first perform an affine transform, that roughly aligns the images, followed by a BSpline transform, that can correct distortions. Not doing the affine first leaves the BSpline transform to do all the coregistration works, which tends to create more artefacts
- Metric: Which metric to use when finding the best registration. 'AdvancedMattesMutualInformation' and 'NormalizedMutualInformation' give decent (and similar) results, other options can be found in the documentation (try HySE.help('CoRegistrationTools.CoRegisterImages') ), but have given worse results.

  
```python

GridSpacing=20
Sigma=1

CoregisteredHypercube, AllTransforms = HySE.CoRegisterHypercube(HypercubeForRegistration, ## Raw hypercube
                                                                Wavelengths_list, ## don't really need this
                                                                Verbose=False, ## Set to True to see all output from SimpleElastix
                                                                Transform='BSplineTransform', ## Transform to use. BSpline best for in vivo data, see documentation for options
                                                                Affine=False, ## Set to true to restrict to Affine transforms
                                                                Blurring=True, ## Set to true for blurring 
                                                                Sigma=Sigma, ## Blurring size
                                                                Mask=Mask_Invert, ## Inficate Mask
                                                                TwoStage=True, ## Set to true for a affine transform followed by BSpline
                                                                Metric='AdvancedMattesMutualInformation', # See documentations for options
                                                                GridSpacing=GridSpacing) ## Grid Spacing

```

### Save Results
And then you might want to calculate the normalised mutual information before and after to estimate the performance of the registration, and save the registered hypercube as a video to see the results.

```python
NMI_Sweep_before = HySE.GetNMI(HypercubeForRegistration)
NMI_Sweep_after = HySE.GetNMI(CoregisteredHypercube)
print(f'Average NMI before : {NMI_Sweep_before[0]:.4f}, Average NMI after : {NMI_Sweep_after[0]:.4f}')

Info = f'Frame{Nframe}_Blurring{Sigma}_AdvMI_TwoStage_GridSpacing{GridSpacing}'
VideoSavingPath = f'{SavingPath}{Name}_{NameSub}_RawFrames/Sweep{Nsweep}_SE_{Info}_NMI{NMI_Sweep_after[0]:.2f}.mp4'

## Whole image
HySE.MakeHypercubeVideo(CoregisteredHypercube, VideoSavingPath, Normalise=True)

## Cropped image
Cropping = 100
VideoSavingPathCropped = VideoSavingPath.replace('.mp4','_Cropped.mp4')
HySE.MakeHypercubeVideo(CoregisteredHypercube[:,Cropping:-1*Cropping,Cropping:-1*Cropping], VideoSavingPathCropped, Normalise=True)

## Not CoRegistered
OrigVideoSavingPath = f'{SavingPath}{Name}_{NameSub}_RawFrames/Sweep{Nsweep}_NoReg.mp4'
HySE.MakeHypercubeVideo(HypercubeForRegistration, OrigVideoSavingPath)
```

### Apply transforms to other data
The transforms optimised during the registration are stored in the AllTransforms list output by HySE.CoRegisterHypercube() function. Each element in the list is a specific transform applied to the associated frame. Each of those transforms can also be saved as .txt files.

To apply the transforms to new frames, use the ApplyTransforms() function:

```python
FramesTransformed = HySE.ApplyTransform(FramesForTransformApply, AllTransforms)
```
The new FramesTransformed array will now be transformed in the same way as the original data.


### Manual Registration
In some cases, SimpleITK is unable to provide decent registration. If the image has some visible features that we are trying to align (particularly lesions), a manual (brute force) registration can be used. For this registration, the user defines fixed points for all images, which are used to compute affine registration. Pop-up windows help to identify those fixed points manually.
The following example shows how to use manual registration.

```python
## Load unregistered data:
HypercubeForRegistration = np.load(HypercubeForRegistration_Path)['arr_0']
AllReflectionsMasks = np.load(AllReflectionsMasks_Path)['arr_0']
EdgeMask = np.load(EdgeMask_Path)['arr_0']

## Run the code to manually indentify fixed points and follow instructinos
Sigma = 2
DeviationThreshold=150
RegHypercube, AllTransforms, RegMask, AllPoints = HySE.CoRegisterHypercubeAndMask_Hybrid(HypercubeForRegistration, 
                                                                                         Wavelengths_list, 
                                                                                         Static_Index=9, 
                                                                                         AllReflectionsMasks=AllReflectionsMasks, 
                                                                                         EdgeMask=EdgeMask, 
                                                                                         InteractiveMasks=True,
                                                                                         Blurring=True, 
                                                                                         Sigma=Sigma, 
                                                                                         DeviationThreshold=DeviationThreshold)
```
At first a window prompting the user to identify fixed points on the static image will open. There is no limit to the number of fixed points, and the colourbar will expand as points are added. Press "z" to remove the latest point, and "p" to toggle the numbering of the points. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/f4863536-f631-4cc1-bbd7-cbaa2edfde81" width="800"/>
</p>

Once all points on the fixed image have been identified, the window can be closed. A new window will then appear, with the static image and already idenfitied fixed points showed on the left frame. The right frame will show the first moving image, where the user must then identify the same fixed points. Like for the static image, pressing "z" will remove the latest point and "p" will toggle the numbering of the points. An automatic warning will apear if the distance between the fixed point in the moving image and the equivalent point in the static image is larger than DeviationThreshold (set to 150 pixels in this example). If this warning appears, it might be that the user has confused points (use "p" to show the points numbers), or that the moving image has enough movement that the same point has moved by more than DeviationThreshold. Once all fixed points have been identified, the window will indicate so and the user may then close the window. A new identical window will then appear, prompting the user to identify fixed points for the second moving image. The same thing will happen untill all moving images (typically 15) have been labelled.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1e329078-526d-4049-94a9-9525ee440777" width="800"/>
</p>


Once all images have been labelled and if the user is confident in their labelling, it is best to save the coordinates for all points for reproducibility:
```python
Npoints = len(AllPoints['fixed_points'])
Info = f'ManualRegistration_{Npoints}points'
PointsSavingPath = f'{SavingPath}{Name}_{NameSub}_RawFrames/Sweep{Nsweep}_Crop{Cropping}_{Info}__Landmarkpoints.npz'
np.savez(f'{PointsSavingPath}', AllPoints, allow_pickle=True)
```

These saved coordinates can then be used to run the registration by adding the AllLandmarkPoints input in the CoRegisterHypercubeAndMask_Manual() function. When AllLandmarkPoints is indicated, the function does not prompt the user to label images but instead performs the registration based on those indicated points.
```python
AllLandmarkPoints_1 = np.load(PointsPath, allow_pickle=True)['arr_0'].item()
RegHypercube, AllTransforms, RegMask, AllPoints = HySE.CoRegisterHypercubeAndMask_Manual(HypercubeForRegistration, 
                                                                                         Wavelengths_list,
                                                                                         Static_Index=9, 
                                                                                         AllReflectionsMasks=AllReflectionsMasks, 
                                                                                         EdgeMask=EdgeMask, 
                                                                                         InteractiveMasks=True, 
                                                                                         Blurring=True, 
                                                                                         Sigma=Sigma, 
                                                                                         DeviationThreshold=DeviationThreshold,
                                                                                         AllLandmarkPoints=AllLandmarkPoints_1)

```

## Help 

A general help function allows to print all the modules and associated functions:
```python
HySE.help()
```

Adding a module in argument allows to print in more details the functinos it contains:
```python
HySE.help('Unmixing')
```

A detailed description of a given function with its inputs and outputs can be accessed by adding the function to the argument:
```python
HySE.help('Unmixing.MakeMixingMatrix')
```

Most functions also allow the input of 'Help=True' as an argument, which will print the same detailed function description. 
The output (0) will match the size of the expected output in order to allow debugging on the go.

```python
HySE.MakeMixingMatrix(_,_,Help=True)
```

</details>

## To do
- [ ] Reference chanel normalisation

## Comments
- If the functions cannote be loaded, try adding an empty \__init__.py file in the same folder
