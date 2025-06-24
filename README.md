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

The co-registration is done with simple elastix / simple itk.

Documentation available here:
https://simpleitk.org/
https://simpleelastix.readthedocs.io/

NB: The simple elastix documentation includes complicated installation instructions that (on MacOS) always crash.
Try first: 
```python
pip install SimpleITK
```

It also requires those additional libraries:
- time
- tqdm

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
Wavelengths_list = np.array([486,572,478,646,584,511,617,540, 643,606,563,498,594,526,630,553])
Wavelengths_list_sorted = np.sort(Wavelengths_list)

## Indicate the mixing matrix that was used in the Arduino code:
Arduino_MixingMatrix = np.array([[0,3,6],
                                 [1,4,7],
                                 [2,5,0],
                                 [3,6,1],
                                 [4,7,2],
                                 [5,0,3],
                                 [6,1,4],
                                 [7,2,5]])

## Locate the data to analyse
DataPath_General = '{Path_to_data}/Data/'

WhiteCalibration_Path = DataPath_General+'2025-03-28_13-24-53_Config1_White_Calib.mp4'
WhiteHySE_Path = DataPath_General+'2025-03-28_13-28-23_Config1_White_HySE.mp4'
MacbethHySE_Path = DataPath_General+'2025-03-28_14-06-19_Config1_Macbeth_HySE_R2DR5.mp4'

Name = '2025-03-28_14-06-19_Config1_Macbeth_HySE_R2DR5'

## Define Saving path:
SavingPath = '/Users/iracicot/Library/CloudStorage/OneDrive-UniversityofCambridge/Data/CCRC/20250328/Results_MultipleSweeps/'
```

### Compute raw arrays

Then raw videos are imported to extract raw (mixed) arrays. 
The code has automatic detections features, but clinical data tends to be too noisy for those to work properly. In these cases, a blind variation allows to input the start position and width of each sweeps.

Mixed clinical data will typically consist of three types of videos:
* WhiteCalibration: This is a dataset obtained by imaging a white reference one wavelength at a time
* WhiteHySE: This is a dataset obtained by imaging a white reference with the mixed wavelengths illumination
* DataHySE: This is a dataset obtained by imaging am object of interest (macbeth chart for tests, or tissue) with the mixed wavelengths illumination


```python
############################
####### 1. White Calibration
############################

## Input the position of the sweep starts (might require iterating)
StartFrames = [148,1066,1984,2901,3820]

## Identify all full sweeps present in the dataset. When using the blind method, use:
##   Blind=True
##   StartFrames=StartFrames
##   PlateauSize={Estimated_Plateau_Size}
EdgePos_WhiteCalib = HySE.FindHypercube(WhiteCalibration_Path, Wavelengths_list, PlateauSize=45, PrintPeaks=False, Blind=True, StartFrames=StartFrames, PeakHeight=0.1, 
                             SaveFig=False, PlotGradient=False, PeakDistance=30, MaxPlateauSize=60)

```
<p align="center">
  <img src="https://github.com/user-attachments/assets/a8f41de6-3ada-44a0-88ae-c6d5dffb5fb8" width="700"/>
</p>

```python

## Compute array from sweeps
Hypercube_WhiteCalib, Dark_WhiteCalib = HySE.ComputeHypercube(WhiteCalibration_Path, EdgePos_WhiteCalib, Wavelengths_list, 
                                                              Order=True, SaveFig=False, SaveArray=False, Help=False, PlotHypercube=False)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/0ce86d23-41f9-496b-aad2-77f85d78d0bc" width="500"/>
</p>

```python

############################
####### 2. White mixed wavelengths
############################

StartFrames = [489, 1012, 1534, 2056]

## Note that the expected PlateauSize will depend on the repeat used during recording
EdgePos_WhiteHySE = HySE.FindHypercube(WhiteHySE_Path, Wavelengths_list, PlateauSize=27, PrintPeaks=False, Blind=True, StartFrames=StartFrames, PeakHeight=0.1, 
                             SaveFig=False, PlotGradient=False, PeakDistance=30, MaxPlateauSize=60)

Hypercube_WhiteHySE, Dark_WhiteHySE = HySE.ComputeHypercube(WhiteHySE_Path, EdgePos_WhiteHySE, Wavelengths_list, 
                                                              Order=False, SaveFig=False, SaveArray=False, Help=False, PlotHypercube=True)


############################
####### 3. Data mixed wavelengths
############################

## This example uses test data from a macbeth chart
StartFrames = [220,567,922,1270,1624,1972,2326] 

EdgePos_MacbethHySE = HySE.FindHypercube(MacbethHySE_Path, Wavelengths_list, PlateauSize=18, PrintPeaks=False, Blind=True, StartFrames=StartFrames, PeakHeight=0.1, 
                             SaveFig=False, PlotGradient=False, PeakDistance=30, MaxPlateauSize=60)

Hypercube_MacbethHySE_avg, Dark_MacbethHySE_avg = HySE.ComputeHypercube(MacbethHySE_Path, EdgePos_MacbethHySE, Wavelengths_list, Buffer=6, Average=True,
                                                                Order=False, Help=False, SaveFig=False, SaveArray=False, PlotHypercube=True)

## In some cases, we might want to keep each sweep individual (instead of averaging them all). This can be done with the 'Average' input:
Hypercube_MacbethHySE_all, Dark_MacbethHySE_all = HySE.ComputeHypercube(MacbethHySE_Path, EdgePos_MacbethHySE, Wavelengths_list, Buffer=6, Average=False,
                                                                Order=False, Help=False, SaveFig=False, SaveArray=False, PlotHypercube=True)

############################
####### 4. Dark
############################

## The function that computes the array (or the raw, mixed hypercube) also outputs a dark computed from the short dark in the middle of
##     each sweeps (between panels 2 and 4). This dark is however not always reliable, and for short repeats is the average of a small number
##     frames. It is preferable to use the long darks between each sweeps to calculate a dark frame, when possible

## In this example, the WhiteCalibration dataset includes an extra plateau at the end of the sweep, when red light was flashed
##     to clearly indicate the end of the sweep. The 'ExtraWav' parameter allows to account for this.
##     The 'Buffer' parameter allows to control how many frames a thrown out on either side of the selection
LongDark = HySE.GetLongDark(WhiteCalibration_Path, EdgePos_WhiteCalib, ExtraWav=1, Buffer=30)

## Plotting the dark can help make sure the estimate is adequate
HySE.PlotDark(LongDark)
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/372215d3-dd7f-4275-864d-54b38a1339c5" width="400"/>
</p>

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
  <img src="https://github.com/user-attachments/assets/5b5fabae-9fd9-4fbb-93a9-d354557b8b1e" width="600"/>
</p>

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
- [ ] Co-registration
   - [ ] Try SimpleElastix with better data
   - [ ] Other algirithm

## Comments
- If the functions cannote be loaded, try adding an empty \__init__.py file in the same folder
