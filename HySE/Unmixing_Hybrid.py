
"""

Functions used for wavelenght unmixing - 3 wav hybrid
 
"""

import numpy as np
# from datetime import datetime
import matplotlib
from matplotlib import pyplot as plt
import inspect
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from scipy.optimize import lsq_linear
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"
# from joblib import Parallel, delayed
# import HySE.Masking
import copy



def GetBlueMatrix_3Wav(GreenMatrix, index_418=7,
                      SameIms=[0,4]):
    Nframes = len(GreenMatrix)
    BlueMatrix = []
    for i in range(0,Nframes):
        line = copy.deepcopy(GreenMatrix[i])
        if i in SameIms:
            BlueMatrix.append(line)
        else:
            line.append(index_418)
            BlueMatrix.append(line)
    return BlueMatrix

def GetShortGreenMatrices_3Wav(GreenMatrix, index_418=7, 
                               indices_4x4=[1,4,5,6], 
                               indices_3x3=[0,2,3]):
    Nframes = len(GreenMatrix)
    matrix_6x6 = []
    matrix_4x4 = []
    matrix_3x3 = []
    for i in range(0,Nframes):
        line = GreenMatrix[i]
        if index_418 in line:
            print(f'Im {i} contains FSK to avoid (418/478). Skipping for 6x6 matrix.')
        else:
            matrix_6x6.append(line)
        if any(x in line for x in indices_4x4):
            if any(x in line for x in indices_3x3):
                print(f'Error: there seems to be a mixing of the indices indicated for the 4x4 and 3x3 submatrices:')
                print(f'  indices_4x4={indices_4x4}')
                print(f'  indices_3x3={indices_3x3}')
            else:
                matrix_4x4.append(line)
        if any(x in line for x in indices_3x3):
            if any(x in line for x in indices_4x4):
                print(f'Error: there seems to be a mixing of the indices indicated for the 4x4 and 3x3 submatrices:')
                print(f'  indices_4x4={indices_4x4}')
                print(f'  indices_3x3={indices_3x3}')
            else:
                matrix_3x3.append(line)
                
    return matrix_6x6, matrix_4x4, matrix_3x3
        
    
    
def GetShortPanelWavs_3Wav(Panel2_Wavelengths, Panel4_Wavelengths, indices_4x4, indices_3x3, index_418):
    P2 = list(Panel2_Wavelengths)
    P4 = list(Panel4_Wavelengths)
    P2.pop(index_418)
    P4.pop(index_418)
    Wavs_6x6_P2 = P2
    Wavs_6x6_P4 = P4
    P2_4x4 = list(Panel2_Wavelengths)
    P4_4x4 = list(Panel4_Wavelengths)
    P2_4x4 = [P2_4x4[x] for x in indices_4x4]
    P4_4x4 = [P4_4x4[x] for x in indices_4x4]
    Wavs_4x4_P2 = P2_4x4
    Wavs_4x4_P4 = P4_4x4
    
    P2_3x3 = list(Panel2_Wavelengths)
    P4_3x3 = list(Panel4_Wavelengths)
    P2_3x3 = [P2_3x3[x] for x in indices_3x3]
    P4_3x3 = [P4_3x3[x] for x in indices_3x3]
    Wavs_3x3_P2 = P2_3x3
    Wavs_3x3_P4 = P4_3x3
    return Wavs_6x6_P2, Wavs_6x6_P4, Wavs_4x4_P2, Wavs_4x4_P4, Wavs_3x3_P2, Wavs_3x3_P4


def generate_local_index_matrix(original_matrix, global_indices_for_subgroup):
    """
    Remaps a section of a global mixing matrix to a local, zero-based index matrix.

    This function is designed to create a "sub-matrix" suitable for unmixing a
    smaller group of wavelengths. It filters the original matrix to include only
    the rows that are exclusively composed of the specified global indices, and then
    converts those global indices to a local 0, 1, 2, ... format.

    Inputs:
        - original_matrix (list of lists): The full mixing matrix containing
          "global" indices (e.g., [[4, 5, 6], [8, 9, 10], [4, 6, 7]]).
        - global_indices_for_subgroup (list): A list of the global indices that
          define the subgroup you want to isolate (e.g., [4, 5, 6, 7]).

    Outputs:
        - sub_matrix (list of lists): A new mixing matrix where the rows from the
          original matrix that fit the subgroup have been re-indexed to be
          zero-based (e.g., [[0, 1, 2], [0, 2, 3]]).

    """
    # Create a mapping from the global index to its new local index (0, 1, 2...)
    # This is the core of the conversion. e.g., {4: 0, 5: 1, 6: 2, 7: 3}
    index_map = {global_idx: local_idx for local_idx, global_idx in enumerate(global_indices_for_subgroup)}

    # Use a set for efficient checking of which indices are valid for our subgroup.
    valid_global_indices = set(global_indices_for_subgroup)

    sub_matrix = []
    # Iterate through each combination (row) in the original matrix
    for original_row in original_matrix:
        # Check if ALL indices in this row belong to our desired subgroup
        if all(idx in valid_global_indices for idx in original_row):
            # If they do, create the new row by looking up each global index
            # in our map to get its new local index.
            new_local_row = [index_map[global_idx] for global_idx in original_row]
            sub_matrix.append(new_local_row)

    return sub_matrix