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

```python
### Green Matrix full 16x16
Title = 'Green Matrix - '+Name
GreenMatrix_7x7 = HySE.MakeMixingMatrix_Flexible(Panel2_Wavelengths, MixingMatrix_Indices, 
                                         Panel4_Wavelengths, MixingMatrix_Indices, 
                                         SaveFig=False, Title=Title, SavingPath=f'{SavingPath}Green_MixingMatrix_{Name}.png')

cond_number = np.linalg.cond(GreenMatrix_7x7)
print("Condition number:", cond_number)


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

# UPDATED UP TO HERE

### Normalisation

The data now needs to be normalised. Several options are possible, all through the same function 'HySE.NormaliseMixedHypercube'. 
Optional arguments allow to control what normalisation is done:
* When 'Dark' is specified, the function with subtract the dark frame input
* When WhiteCalibration is specified, the function with normalise (divide) each frame by the equivalent white calibration frame.

The function also estimates a mask, from the white calibration when possible and from the data otherwise, that masks the dark corners from the endoscopic data.

Depending on the selected normalisation, sometimes the automatic plotting range is inadequate. The 'vmax' option allows to manually adjust the upper scale.


```python


############################
####### 6. Normalise Data
############################

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
  <img src="https://github.com/user-attachments/assets/5b5fabae-9fd9-4fbb-93a9-d354557b8b1e" width="750"/>
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
FramesTransformed = ApplyTransform(FramesForTransformApply, AllTransforms)
```
The new FramesTransformed array will now be transformed in the same way as the original data.

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
