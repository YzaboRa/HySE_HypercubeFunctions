# HySE_HypercubeFunctions
Functions to compute hypercube from endoscope videos 

## Installation
Python 3 code, requires the following packaging:

- numpy
- opencv-python
- DateTime
- scipy
- matplotlib
- ffmpeg

## How to use
```python
## Import functions
import sys
PathToFunctions = '{Path to where the HySE_HypercubeFunctions.py code is located}'
sys.path.append(PathToFunctions)
from ScopeAnalysisFunctions import *

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
EdgePos = FindHypercube(vidPath, Wavelengths_list, PeakHeight=0.045)

## Will output something like this:
<img width="452" alt="FindHypercubeOutput" src="https://github.com/user-attachments/assets/b6d45c7a-b74b-455e-97e9-ff4649ad153e">



## Once the sweeps have been properly identified, compute the hypercube
Hypercube, Dark = ComputeHypercube(vidPath, EdgePos, Wavelengths_list, Name=Name)
## Will output something like this:
<img width="304" alt="ComputeHypercubeOutput" src="https://github.com/user-attachments/assets/afd3fbb1-79c5-4c69-9a6d-0dc0e0fe4a1f">


## Obtain the white reference hypercube to be used to normalise the illumination profile
## By performing the same steps on the white reference data

Name_White = 'White_5x_OD_2cm'
vidPath_White = SavingPath+Name_White+'.mp4'
##   # Note that the parameters are different for the white reference in this example because the repeat number is different
EdgePos_White = FindHypercube(vidPath_White, Wavelengths_list, MaxSize=60, DarkMin=150, PeakHeight=0.1, PlateauSize=54)

## When happy with the identified sweeps
Hypercube_White, Dark_White = ComputeHypercube(vidPath_White, EdgePos_White, Wavelengths_list, Name=Name)

## And finally normalise the hypercube
HypercubeNormalised = NormaliseHypercube(Hypercube, Hypercube_White, Dark_White, Wavelengths_list)

## Will output something like this:
<img width="296" alt="NormaliseHypercubeOutput" src="https://github.com/user-attachments/assets/42648e4a-8a94-481b-9727-0c1ae76998be">


## The hypercubes (saved as npy files) can be visualised with the Hypercube visualiser
```


## Comments
- The data and plots will be saved where this main code is located (not where the 
- If the functions cannote be loaded, try adding an empty __init__.py file in the same folder
