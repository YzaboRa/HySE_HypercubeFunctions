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
```python
## Import functions
import sys
PathToFunctions = '{Path to where the HySE_HypercubeFunctions files are located}'
sys.path.append(PathToFunctions)
import Hyse

## Indicate which wavelengths were used (Panel4, Panel2), in nm
Wavelengths_list = np.array([486,572,478,646,584,511,617,540, 643,606,563,498,594,526,630,553])

## Locate the data to analyse
DataPath = '{Path to the data, e.g. OneDrive folder}'
Name = 'USAF_3x_OD_2cm'
vidPath = DataPath+Name+'.mp4'

## Find the positions of each wavelengths in the data trace
## Start with Help=True to have a list and descriptions of all the
## parameters to play with, and then with PlotGradient=True
## to help optimising
EdgePos = HySE.FindHypercube(vidPath, Wavelengths_list, PeakHeight=0.045)

## We will need to to do the same thing for the white reference data:
EdgePosWhite = HySE.FindHypercube(vidPathWhite, Wavelengths_list, PlotGradient=True)

## We can also at this stage compute the average dark frame between the sweeps:
Dark = HySE.GetDark(vidPath, EdgePos)
## Optional inputs include Buffer=6, DarkRepeat=3, CropImDimensions=[x0,xe, y0,ye]

```
Once the sweeps have been properly identified, we can compute the hypercube. There are several options.

### Option 1: Co-registration with Normalisation and masking 

This option offers to mask in each frame the regions where there is not enoug light (as determined with the white reference) 
and those where there is too much (for example, to get rid of specular reflections).
Each frame is then normalised according to the white reference and then co-registered.
The co-registration can either be done according to a fixed static image (defined by wavelength), or a rolling method can be implemented
where the co-registration is propagated outwards from the static image. This is an attempt to limit drastic changes in the relative intensities 
between the static and moving image, which the co-registration algorithm doesn't always handle properly. 

Note that the rolling co-registration method, because it uses previously co-registered images as static images, can propagate and 
amplify distortions. Use with care.

```python
## We need to identify which sweeps to analyse (the option to do them all at once is not yet available):
Nsweep = 3
NsweepWhite = 3

## And then we can import the full data for those selected sweep:
DataSweep = HySE.GetSweepData_FromPath(vidPath, EdgePos, Nsweep)
DataSweepWhite = HySE.GetSweepData_FromPath(vidPathWhite, EdgePosWhite, NsweepWhite)

## We can average the white reference for the normalisation by computing the hypercube:
HypercubeWhite, _ = HySE.ComputeHypercube(vidPathWhite, EdgePosWhite, Wavelengths_list)

## Defining Cutoffs for Masking:
HC = 255 ## HighCutoff (saturation)
LC = 0.5 ## LowCutoff (if there is not enough light -> amplify noise)

## Now we have everything we need to compute the normalise, masked hypercube
## If required, Help=True prints a detailed description of both functions

## For the standard implementation:
Hypercube = HySE.SweepCoRegister_MaskedWithNormalisation(DataSweep, HypercubeWhite, Dark, Wavelengths_list, ImStatic_Index=7, LowCutoff=LC, HighCutoff=HC)

## And for the rolling co-registration implentation:
Hypercube = HySE.SweepRollingCoRegister_MaskedWithNormalisation(DataSweep, HypercubeWhite, Dark_Hypercube, Wavelengths_list, ImStatic_Index=7, LowCutoff=LC, HighCutoff=HC)

## Can plot result
HySE.PlotHypercube(Hypercube, Wavelengths=Wavelengths_list, SavePlot=False)

```

### Option 2: Co-registration only

```python
## Select the sweep
Nsweep = 5
## Define where and how the data will be saved
SP = f'{SavingPath}{Name}_NSweep{Nsweep}'

Hypercube = HySE.GetCoregisteredHypercube(vidPath, EdgePos, Nsweep, Wavelengths_list, ImStatic_Plateau=1, ImStatic_Index=8, Buffer=6, SavingPath=SP)
## Optional inputs include Plot_PlateauList='None'/'All'/[1,2, ..], Plot_Index=7, PlotDiff=True


```

### Option 3: Normalisation only  (no co-registration, no masking)

Alternatively, if coregistration is not required, the hypercube can be computed in the following way: 

```python
Hypercube, Dark = HySE.ComputeHypercube(vidPath, EdgePos, Wavelengths_list, Name=Name)
Hypercube_White, Dark_White = HySE.ComputeHypercube(vidPath_White, EdgePos_White, Wavelengths_list, Name=Name)

## And then normalised
HypercubeNormalised = HySE.NormaliseHypercube(vidPath, Hypercube, Hypercube_White, Dark_White, Wavelengths_list)

```
The hypercubes and list of wavelengths (saved as npy files) can be visualised with the HypercubeVisualiser.
Plots and videos can also be generated the following way:

```python
## Define where and how the data will be saved
SP = f'{SavingPath}{Name}_NSweep{Nsweep}'
## Plot
HySE.PlotHypercube(Hypercube, Wavelengths=Wavelengths_list)
## Make video
HySE.MakeHypercubeVideo(Hypercube, SP)
```


Some functions have a specific help flag that print a full description of inputs, outputs and constraints (Help=True). 
A succint general help statement listing optional inputs and default values can be obtained for all functions by executing the following:

```python
help(HySE.FUNCTION)
```


### Here are some examples of outputs:

Output from FindHypercube:
<img width="452" alt="FindHypercubeOutput" src="https://github.com/user-attachments/assets/b6d45c7a-b74b-455e-97e9-ff4649ad153e">

Output from ComputeHypercube:
<img width="304" alt="ComputeHypercubeOutput" src="https://github.com/user-attachments/assets/afd3fbb1-79c5-4c69-9a6d-0dc0e0fe4a1f">

Output from NormaliseHypercube:
<img width="296" alt="NormaliseHypercubeOutput" src="https://github.com/user-attachments/assets/42648e4a-8a94-481b-9727-0c1ae76998be">

## To do
- [ ] Mask images
    - [x] Mask specular reflections
    - [x] Mask areas/edges missing because of movement (co-registration)
    - [ ] Mask images in plots
    - [ ] Implement option to automatically asjust masking for specular reflection
- [x] Add function to brute force plateau detection with just plateau expected size (for when automatic detection doesn't work)
- [x] Add rotating co-registration (set static image as co-registrated image with closest wavelength, to avoid extra distortions)
- [ ] Co-register + normalise multiple sweeps (combine/average frames)
- [ ] Fine-tune normalisation to account for intensity oscillations (3-frame cycle)

## Comments
- If the functions cannote be loaded, try adding an empty \__init__.py file in the same folder
