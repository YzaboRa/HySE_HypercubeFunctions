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

The code was written to be run in a jupyter notebook. Including
```python
%matplotlib ipympl
```
At the start of the notebook enables interactive figures.

## How to use

### Import parameters
The pipeline must first be setup by importing the functions and setting up data and saving parameters:

```python
## Import functions
import sys
PathToFunctions = '{Path to where the HySE_HypercubeFunctions files are located}'
sys.path.append(PathToFunctions)
import Hyse

## Indicate which wavelengths were used (Panel4, Panel2), in nm
## 3 wav split hybrid, wavelengths 1:
##                     0.   1.   2.   3.   4.   5.   6.   7
Panel2_Wavelengths = [625, 606, 584, 560, 543, 511, 486, 418]
Panel4_Wavelengths = [646, 617, 594, 576, 569, 526, 503, 478]

Wavelengths_list = np.array(Panel2_Wavelengths+Panel4_Wavelengths)
Wavelengths_list_sorted = np.sort(Wavelengths_list)

## Indicate the mixing matrix that was used in the Arduino code:
Arduino_MixingMatrix = [[4, 5, 6],
                        [1, 4, 6],
                        [1, 5, 6],
                        [1, 4, 5],
                        [4, 5, 6, 7],
                        [2, 3],
                        [0, 3],
                        [0, 2]]

## Locate the data to analyse
DataPath_General = '{Path_to_data}/Data/'


DataPath = DataPath_General+'{Video_Name}.mp4'
Name = '3Wavs1_StaticMacbeth'

## Define Saving path:
SavingPath = '{Path_to_save}'

```

### Dark
Use a video of dark frames to calculate the dark.

```python
LongDark = HySE.GetDark_WholeVideo(DarkPath) #, CropImDimensions=CropImDimensions
HySE.PlotDark(LongDark)
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/372215d3-dd7f-4275-864d-54b38a1339c5" width="400"/>
</p>


### Trace

Identify sweeps by plotting the trace to find at which frames each one starts. Updated to keep RGB chanels distinct.

```python
## Start with:
StartFrames = []
## And then populate the position of the first green frame for each sweep:
StartFrames = [160, 334, 508, 683]

EdgePos_Data = HySE.FindHypercube_RGB(DataPath, PlateauSize=9, StartFrames=StartFrames, SaveFig=False, MaxPlateauSize=20, fps=60)
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/7fd87cd3-ab97-457a-9f21-80580b51f00e" width="800"/>
</p>
### Extract Frames
Knowing where each sweep is, we can now extract the right frames for each plateau (and remove buffer frames)
```python
Buffer = 3
Frames, RGB_Dark = HySE.ComputeHypercube_RGB(DataPath, EdgePos_Data, Buffer=Buffer, BlueShift=1, SaveArray=False)
```

### Dark Subtraction
Subtract dark from all frames
```python
(Nsweeps, Nwav, Nframes, Y, X, _) = Frames.shape

## BGR 
Frames_Blue = Frames[:,:,:,:,:,0]
Frames_Green = Frames[:,:,:,:,:,1]
Frames_Red = Frames[:,:,:,:,:,2]

# NormaliseMixedHypercube
Frames_BlueD, MaskB = HySE.NormaliseMixedHypercube(Frames_Blue, Dark=LongDark, SaveFigure=False, Plot=False)
Frames_GreenD, MaskG = HySE.NormaliseMixedHypercube(Frames_Green, Dark=LongDark, SaveFigure=False, Plot=False)
Frames_RedD, MaskR = HySE.NormaliseMixedHypercube(Frames_Red, Dark=LongDark, SaveFigure=False, Plot=False)
```

### Average over all Frames
For all sweeps, all frames. Or select a single sweep, or only one frame.
```python
### JUST GREEN
FramesCombined_Green = Frames_GreenD_avg
FramesCombined_Blue = Frames_BlueD_avg
FramesCombined_Red = Frames_RedD_avg

# ### OR JUST GREEN & Individual Sweeps
NSweep = 2
FramesCombined_Green = Frames_GreenD_avg[NSweep,:,:,:]
FramesCombined_Blue = Frames_BlueD_avg[NSweep,:,:,:]
FramesCombined_Reference = Frames_RedD_avg[NSweep,:,:,:]
```

### Normalisation
<details>
<summary>⚠️ Outdated section (click to expand)</summary>

N.B.: This section needs updating to incorporate using the reference channel for spatial normalisation. The names used in this section might not reflect those earlier/later in this file. Normalisation can be skipped.

The data now needs to be normalised. Several options are possible, all through the same function 'HySE.NormaliseMixedHypercube'. 
Optional arguments allow to control what normalisation is done:
* When 'Dark' is specified, the function with subtract the dark frame input
* When WhiteCalibration is specified, the function with normalise (divide) each frame by the equivalent white calibration frame.

The function also estimates a mask, from the white calibration when possible and from the data otherwise, that masks the dark corners from the endoscopic data.

Depending on the selected normalisation, sometimes the automatic plotting range is inadequate. The 'vmax' option allows to manually adjust the upper scale.


```python
##The General function for normalising the data is the following:
Hypercube_MacbethHySE_avg_ND, Mask_avg = HySE.NormaliseMixedHypercube(Hypercube_MacbethHySE_avg, Dark=LongDark, WhiteCalibration=Hypercube_WhiteHySE, Wavelengths_list=Wavelengths_list,
                                                           SaveFigure=False, SavingPath=SavingPath+Name,)
Hypercube_MacbethHySE_avg_N, _ = HySE.NormaliseMixedHypercube(Hypercube_MacbethHySE_avg, WhiteCalibration=Hypercube_WhiteHySE, Wavelengths_list=Wavelengths_list,
                                                           SaveFigure=True, SavingPath=SavingPath+Name)
Hypercube_MacbethHySE_avg_D, _ = HySE.NormaliseMixedHypercube(Hypercube_MacbethHySE_avg, Dark=LongDark, Wavelengths_list=Wavelengths_list,
                                                           SaveFigure=False, SavingPath=SavingPath+Name, vmax=80)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/eb7f9bdb-0ee8-4985-a2ab-d9e3f19d4f36" width="500"/>
</p>


```python
## Using the non averaged dataset:
Hypercube_MacbethHySE_all_ND, Mask_all = HySE.NormaliseMixedHypercube(Hypercube_MacbethHySE_all, Dark=LongDark, WhiteCalibration=Hypercube_WhiteHySE, Wavelengths_list=Wavelengths_list,
                                                           SaveFigure=False, SavingPath=SavingPath+Name)
## Or a single individual sweep:
Hypercube_MacbethHySE_1_ND, _ = HySE.NormaliseMixedHypercube(Hypercube_MacbethHySE_all[0,:,:,:], Dark=LongDark, WhiteCalibration=Hypercube_WhiteHySE, Wavelengths_list=Wavelengths_list,
                                                           SaveFigure=True, SavingPath=SavingPath+Name)

```
</details>
  
### Mask
Only really important if doing co-registration
```python
_, Mask = HySE.NormaliseMixedHypercube(Frames_GreenD[0,:,0,:,:], Dark=LongDark, Wavelengths_list=Wavelengths_list, 
                                       SaveFigure=False, SavingPath=SavingPath+Name, vmax=160, Plot=False)

EdgeMask = HySE.GetBestEdgeMask(Mask)
```

### Mixing Matrices
First define mixing matrices.
The hybrid mixing method involves several matrices. Currently not everything is automated, and some submatrices need to be input manually. Further updates should address this issue.

```python

### Blue Matrix full 16x16
Title = 'Blue Matrix - '+Name
BlueMatrix = HySE.MakeMixingMatrix_Flexible(Panel2_Wavelengths, BlueMatrix_Indices, 
                                         Panel4_Wavelengths, BlueMatrix_Indices, 
                                         SaveFig=False, Title=Title, SavingPath=f'{SavingPath}Blue_MixingMatrix_{Name}.png')

cond_number = np.linalg.cond(BlueMatrix)
print("Condition number:", cond_number)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/49fd5867-b39b-40b0-8cfc-38f39294e873" width="500"/>
</p>

```python
### Green Matrix full 16x16
Title = 'Green Matrix - '+Name
GreenMatrix_7x7 = HySE.MakeMixingMatrix_Flexible(Panel2_Wavelengths, MixingMatrix_Indices, 
                                         Panel4_Wavelengths, MixingMatrix_Indices, 
                                         SaveFig=False, Title=Title, SavingPath=f'{SavingPath}Green_MixingMatrix_{Name}.png')

cond_number = np.linalg.cond(GreenMatrix_7x7)
print("Condition number:", cond_number)

```

<p align="center">
  <img src="https://github.com/user-attachments/assets/6b4f70ad-33ce-4103-be13-c6dd77184728" width="500"/>
</p>

```python
### Short green Matrix 14x14 (remove frames with weak wavelengths)
print(f'GREEN SHORT MIXING MATRIX')
Title = 'Green Matrix 6x6 - '+Name

GreenMatrix_6x6_sub_Indices = generate_local_index_matrix(GreenMatrix_6x6_Indices, indices_6x6)


GreenMatrix_6x6 = HySE.MakeMixingMatrix_Flexible(Wavs_6x6_P2, GreenMatrix_6x6_sub_Indices, 
                                         Wavs_6x6_P4, GreenMatrix_6x6_sub_Indices, 
                                         SaveFig=False, Title=Title, SavingPath=f'{SavingPath}GreenMatrix_6x6_{Name}.png')

cond_number = np.linalg.cond(GreenMatrix_6x6)
print("Condition number:", cond_number)


### Sub green mixing matrices (7x7)
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
Sub1_indices_frames = [0,1,2,3,5,6,7]
Sub2_indices_frames = [8,9,10,11,13,14,15]

Title='Green SubMatrices'

HySE.PlotMixingMatrix(MixingMatrix_Sub1, Wavelengths_Sub1, Title, '')


### Sub Sub green mixing matrices (3x3 and 4x4)

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

Title='Green SubSubMatrix1A'
PlotMixingMatrix(MixingMatrix_SubSub1A, MixingMatrix_SubSub1A, Title, '')

Title='Green SubSubMatrix1B'
PlotMixingMatrix(MixingMatrix_SubSub1B, MixingMatrix_SubSub1B, Title, '')

```

<p align="center">
  <img src="https://github.com/user-attachments/assets/650e1403-c03a-44b0-9424-341d801b709a" width="850"/>
</p>

### Unmixing
First select the appropriate frames to unmix each subset of the dataset:
```python
## If we only have indices only for 8x8 logic (instead of the full 16x16
## Use the omit_frames() function with propagates to the rest of the dataset
Frames_6x6 = HySE.omit_frames(FramesCombined_Green_N, indices_6x6)

## If indices are input manually for the whole dateset, use them directly
## Working with only the larger submatrices (empirically: better results)
Frames_Sub1 = FramesCombined_Green_N[Sub1_indices_frames]
Frames_Sub2 = FramesCombined_Green_N[Sub2_indices_frames]

## Or working with the smallest submatrices (empirically: worse results)
Frames_SubSub1A = FramesCombined_Green_N[SubSub1A_indices_frames]
Frames_SubSub1B = FramesCombined_Green_N[SubSub1B_indices_frames]
Frames_SubSub2A = FramesCombined_Green_N[SubSub2A_indices_frames]
Frames_SubSub2B = FramesCombined_Green_N[SubSub2B_indices_frames]
```

Then unmixing can be done. 
There are 3 functions in this package that can be used for umxing:
- UnmixData(): Basic matrix inversion using least squares
- UnmixDataNNLS(): Matrix inversition using non-negative least squares (NNLS), with noise filtering and rejection of pixels with negative or invalid intensity
- UnmixDataSmoothNNLS(): Matrix inversion using NNLS with with noise filtering and pixel rejection, imposing smoothness (regulation set weighted by lambda)
- UnmixDataSmoothNNLSPrior(): Matrix inversino using NNLS with two regulators, smoothnes (weighted by lambda_smooth) and a known spectral prior (weighted by lambda_prior)

The UnmixDataSmoothNNLS() function generally leads to the best results, with low lambda (~0.1)
```python

## Blue Matrix
t0 = time.time()
Unmixed_Blue = HySE.UnmixDataSmoothNNLS(FramesCombined_Blue_N, BlueMatrix, lambda_smooth=0.1)
t1 = time.time()
print(f'\nFull Blue - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix

## Green Matrix, all frames:
t0 = time.time()
Unmixed_Green_7x7 = HySE.UnmixDataSmoothNNLS(FramesCombined_Green_N, GreenMatrix_7x7, lambda_smooth=0.1)
t1 = time.time()
print(f'Full Green - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, 6x6:
t0 = time.time()
Unmixed_Green_6x6 = HySE.UnmixDataSmoothNNLS(Frames_6x6, GreenMatrix_6x6, lambda_smooth=0.1)
t1 = time.time()
print(f'Green 6x6 - Smooth NNLS unmixing took {t1-t0:.0f}s')


## Green Matrix, Sub1:
t0 = time.time()
Unmixed_Green_Sub1 = HySE.UnmixDataSmoothNNLS(Frames_Sub1, MixingMatrix_Sub1, lambda_smooth=0.1)
t1 = time.time()
print(f'Green Sub1 - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, Sub2:
t0 = time.time()
Unmixed_Green_Sub2 = HySE.UnmixDataSmoothNNLS(Frames_Sub2, MixingMatrix_Sub2, lambda_smooth=0.1)
t1 = time.time()
print(f'Green Sub2 - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, SubSub1A:
t0 = time.time()
Unmixed_Green_SubSub1A = HySE.UnmixDataSmoothNNLS(Frames_SubSub1A, MixingMatrix_SubSub1A, lambda_smooth=0.1)
t1 = time.time()
print(f'Green SubSub1A - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, SubSub1B:
t0 = time.time()
Unmixed_Green_SubSub1B = HySE.UnmixDataSmoothNNLS(Frames_SubSub1B, MixingMatrix_SubSub1B, lambda_smooth=0.1)
t1 = time.time()
print(f'Green SubSub1B - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, SubSub2A:
t0 = time.time()
Unmixed_Green_SubSub2A = HySE.UnmixDataSmoothNNLS(Frames_SubSub2A, MixingMatrix_SubSub2A, lambda_smooth=0.1)
t1 = time.time()
print(f'Green SubSub2A - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, SubSub2B:
t0 = time.time()
Unmixed_Green_SubSub2B = HySE.UnmixDataSmoothNNLS(Frames_SubSub2B, MixingMatrix_SubSub2B, lambda_smooth=0.1)
t1 = time.time()
print(f'Green SubSub2B - Smooth NNLS unmixing took {t1-t0:.0f}s')

```

The results obtained from the submatrices need to be recombined so that a full spectra is obtained:

```python
# Larger submatrices:
Unmixed_GreenSub12_Combined, Wavs_Sub12_Combined = HySE.combine_hypercubes(Unmixed_Green_Sub1, Unmixed_Green_Sub2, 
                                                                           Wavelengths_Sub1, Wavelengths_Sub2)

## Smaller submatrices:
Unmixed_GreenSubSub1_Combined, Wavs_SubSub1_Combined = HySE.combine_hypercubes(Unmixed_Green_SubSub1A, Unmixed_Green_SubSub1B, 
                                                                               Wavelengths_SubSub1A, Wavelengths_SubSub1B)
Unmixed_GreenSubSub2_Combined, Wavs_SubSub2_Combined = HySE.combine_hypercubes(Unmixed_Green_SubSub2A, Unmixed_Green_SubSub2B, 
                                                                               Wavelengths_SubSub2A, Wavelengths_SubSub2B)

Unmixed_GreenSubSubAll_Combined, Wavs_SubSubAll_Combined = HySE.combine_hypercubes(Unmixed_GreenSubSub1_Combined, Unmixed_GreenSubSub2_Combined, 
                                                                                   Wavs_SubSub1_Combined, Wavs_SubSub2_Combined)

```

And save for future references/easier registration.

```python
print(f'Combined Wavelengths - Larger submatrices:')
## Sanity check to make sure the combined wavelengths correspond to the right ones, ordered
print(Wavs_Sub12_Combined) 
np.savez(SavingPath+'HybridWav3_GreenCombined_Sub12.npz', Unmixed_GreenSub12_Combined)
np.savez(SavingPath+'HybridWav3_GreenCombined_Sub12_wavelengths.npz', Wavs_Sub12_Combined)

print(f'Combined Wavelengths - Smaller submatrices:')
print(Wavs_SubSubAll_Combined)
np.savez(SavingPath+'HybridWav3_GreenCombined_SubSubAll.npz', Unmixed_GreenSubSubAll_Combined)
np.savez(SavingPath+'HybridWav3_GreenCombined_SubSubAll_wavelengths.npz', Wavs_SubSubAll_Combined)


```


# UPDATED UP TO HERE


### Unmixing

Once this is done, we can then move on to the actual unmixing:

```python
############################
####### 5. Mixing Matrix
############################

## Compute the mixing matrix
MixingMatrix = HySE.MakeMixingMatrix(Wavelengths_list, Arduino_MixingMatrix, Help=False)
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/e71b8693-c12b-4a80-8303-4bda44a04c0e" width="400"/>
</p>

```python

############################
####### 6. Unmix the data
############################

Unmixed_Hypercube_MacbethHySE_avg_ND = HySE.UnmixData(Hypercube_MacbethHySE_avg_ND, MixingMatrix)
Unmixed_Hypercube_MacbethHySE_avg_N = HySE.UnmixData(Hypercube_MacbethHySE_avg_N, MixingMatrix)
Unmixed_Hypercube_MacbethHySE_avg_D = HySE.UnmixData(Hypercube_MacbethHySE_avg_D, MixingMatrix)

Unmixed_Hypercube_MacbethHySE_all_ND = HySE.UnmixData(Hypercube_MacbethHySE_all_ND, MixingMatrix)
Unmixed_Hypercube_MacbethHySE_1_ND = HySE.UnmixData(Hypercube_MacbethHySE_1_ND, MixingMatrix)

## We can plot this hypercube for visualisation:
SP = f'{SavingPath}{Name}_UnmixedND_avg.png'
HySE.PlotHypercube(Unmixed_Hypercube_MacbethHySE_avg, Wavelengths=Wavelengths_list_sorted, SameScale=False, Masks=Mask_avg, SavePlot=False, SavingPathWithName=SP) #vmax=0.5,
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/904991ed-4660-4f37-a5de-291f8ef8407a" width="500"/>
</p>

### Saving
Don't forget to save the unmixed hypercubes! The functions may save figures, but you need to save the array itself:

```python

np.savez(f'{SavingPath}{Name}_UnmixedHypercube_MacbethHySE_avg_ND.npz', Unmixed_Hypercube_MacbethHySE_avg_ND)
np.savez(f'{SavingPath}{Name}_UnmixedHypercube_MacbethHySE_avg_N.npz', Unmixed_Hypercube_MacbethHySE_avg_N)
np.savez(f'{SavingPath}{Name}_UnmixedHypercube_MacbethHySE_avg_D.npz', Unmixed_Hypercube_MacbethHySE_avg_D)

np.savez(f'{SavingPath}{Name}_UnmixedHypercube_MacbethHySE_all_ND.npz', Unmixed_Hypercube_MacbethHySE_all_ND)
np.savez(f'{SavingPath}{Name}_UnmixedHypercube_MacbethHySE_1_ND.npz', Unmixed_Hypercube_MacbethHySE_1_ND)

## The Hypercube Visualiser GUI will also require a list of the wavelengths:
np.savez(f'{SavingPath}{Name}_SortedWavelengths.npz', Wavelengths_list_sorted)

```

## Optional Functions

### Macbeth Colour Chart analysis

When imaging a Macbeth colour chart, it is helpful to look at the spectra from each patch.
Certain functions are designed to make this task easier.

```python

############################
####### A.1 Location of ground truth files
############################

MacbethPath = 'MacBeth/Micro_Nano_CheckerTargetData.xls'

Macbeth_sRGBPath = 'MacBeth/Macbeth_sRGB.xlsx'
Macbeth_AdobePath = 'MacBeth/Macbeth_Adobe.xlsx'

MacBethSpectraData = np.array(pd.read_excel(MacbethPath, sheet_name='Spectra'))
MacBethSpectraColour = np.array(pd.read_excel(MacbethPath, sheet_name='Color_simple'))

MacBeth_RGB = np.array(pd.read_excel(Macbeth_AdobePath))
## Adobe RGB
MacBethSpectraRGB = np.array([MacBethSpectraColour[:,0], MacBethSpectraColour[:,7], MacBethSpectraColour[:,8], MacBethSpectraColour[:,9]])


############################
####### A.2 Crop around macbeth chart
############################

xs, xe = 170,927
ys, ye = 310,944

macbeth = Unmixed_Hypercube_MacbethHySE_avg_ND[-1,ys:ye,xs:xe]

plt.close()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5))
ax.imshow(macbeth, cmap='gray')
plt.tight_layout()
plt.show()


############################
####### A.3 Identify where each patch is located
############################

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
############################
####### A.4 Compute the spectra for each patch
############################

CropCoordinates = [xs, xe, ys, ye]

PatchesSpectra_MacbethHySE_avg_ND = HySE.GetPatchesSpectrum(Unmixed_Hypercube_MacbethHySE_avg_ND, Sample_size, Positions, CropCoordinates)
PatchesSpectra_MacbethHySE_avg_N = HySE.GetPatchesSpectrum(Unmixed_Hypercube_MacbethHySE_avg_N, Sample_size, Positions, CropCoordinates)
PatchesSpectra_MacbethHySE_avg_D = HySE.GetPatchesSpectrum(Unmixed_Hypercube_MacbethHySE_avg_D, Sample_size, Positions, CropCoordinates)

PatchesSpectra_MacbethHySE_all_ND = HySE.GetPatchesSpectrum(Unmixed_Hypercube_MacbethHySE_all_ND, Sample_size, Positions, CropCoordinates)
PatchesSpectra_MacbethHySE_1_ND = HySE.GetPatchesSpectrum(Unmixed_Hypercube_MacbethHySE_1_ND, Sample_size, Positions, CropCoordinates)

############################
####### A.5 Plot
############################

## Indicate which spectra to plot
##     If plotting more than one, create a list with all the spectra
PatchesToPlot = [PatchesSpectra_MacbethHySE_avg_ND, PatchesSpectra_MacbethHySE_all_ND, PatchesSpectra_MacbethHySE_1_ND]
## For a more descriptive plot, define short labels for each spectra in the list
Labels = ['avg then unmix', 'unmix then avg', 'one sweep']
## If saving the figure, define a saving path
Name_ToSave = f'{SavingPath}{Name}_UnmixingComparison_ND'

HySE.PlotPatchesSpectra(PatchesToPlot, Wavelengths_list_sorted, MacBethSpectraData, MacBeth_RGB, Name, PlotLabels=Labels)#, SavingPath=Name_ToSave)
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/5b5fabae-9fd9-4fbb-93a9-d354557b8b1e" width="800"/>
</p>



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
Co-registration performs better when the sharp black edges from the endoscopy monitor display are masked

```python
## Use the data iself, or any normalisation, to estimate the sharp edges mask (anything works)
_, Mask = HySE.NormaliseMixedHypercube(Hypercube_Lesion1_all[2,:,:,:], Dark=LongDark, Wavelengths_list=Wavelengths_list, 
                                       SaveFigure=False, SavingPath=SavingPath+Name, vmax=160, Plot=False)

## SimpleITK expects a mask with the inverse logic
Mask_Invert = np.invert(Mask)
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

## To do
- [ ] Update masking of images
   - [ ] Specular reflections
   - [ ] Low intensity areas
- [x] Spectra only normalisation
- [ ] Try different unmixing algorithms
   - [ ] Non-Negative Least Squares (NNLS)
   - [ ] Weighted Least Squares (WLS)
   - [ ] Tikhonov Regularization (Ridge Regression)
   - [ ] Sparse Unmixing (Lasso)
- [ ] Co-registration
   - [ ] Try SimpleElastix with better data
   - [ ] Other algorithm

## Comments
- If the functions cannote be loaded, try adding an empty \__init__.py file in the same folder
