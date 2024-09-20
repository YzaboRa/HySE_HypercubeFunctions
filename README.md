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


## Once the sweeps have been properly identified, compute the hypercube
## Select the sweep (currently can only co-register a single sweep)
Nsweep = 5
## Define where and how the data will be saved
SP = f'{SavingPath}{Name}_NSweep{Nsweep}'

Hypercube = HySE.GetCoregisteredHypercube(vidPath, EdgePos, Nsweep, Wavelengths_list, ImStatic_Plateau=1, ImStatic_Index=8, Buffer=6, SavingPath=SP)
## Optional inputs include Plot_PlateauList='None'/'All'/[1,2, ..], Plot_Index=7, PlotDiff=True


## Plot
HySE.PlotHypercube(Hypercube, Wavelengths=Wavelengths_list)
## Make video
HySE.MakeHypercubeVideo(Hypercube, SP)


## Get Dark:
Dark = HySE.GetDark(vidPath, EdgePos)
## Optional inputs include Buffer=6, DarkRepeat=3, CropImDimensions=[x0,xe, y0,ye]


## Obtain the white reference hypercube to be used to normalise the illumination profile
## By performing the same steps on the white reference data

Name_White = 'White_5x_OD_2cm'
vidPath_White = SavingPath+Name_White+'.mp4'
##   # Note that the parameters are different for the white reference in this example because the repeat number is different
EdgePos_White = HySE.FindHypercube(vidPath_White, Wavelengths_list, MaxSize=60, DarkMin=150, PeakHeight=0.1, PlateauSize=54)

```
The hypercubes (saved as npy files) can be visualised with the Hypercube visualiser

Alternatively, if coregistration is not required, the hypercube can be computed in the following way: 

```python

Hypercube, Dark = HySE.ComputeHypercube(vidPath, EdgePos, Wavelengths_list, Name=Name)
Hypercube_White, Dark_White = HySE.ComputeHypercube(vidPath_White, EdgePos_White, Wavelengths_list, Name=Name)

## And then normalised
HypercubeNormalised = HySE.NormaliseHypercube(vidPath, Hypercube, Hypercube_White, Dark_White, Wavelengths_list)

## The hypercubes (saved as npy files) can be visualised with the Hypercube visualiser
```

Some functions have a specific help flag that print a full description of inputs, outputs and constraints. 
A general help statement can be obtained for all functions by executing the following:
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
- [ ] Add function to brute force plateau detection with just plateau expected size (for when automatic detection doesn't work)
- [x] Add rotating co-registration (set static image as co-registrated image with closest wavelength, to avoid extra distortions)
- [ ] Co-register + normalise multiple sweeps (combine/average frames)
- [ ] Fine-tune normalisation to account for intensity oscillations (3-frame cycle)
- [ ] Investitgate dark subtraction

## Comments
- If the functions cannote be loaded, try adding an empty \__init__.py file in the same folder
