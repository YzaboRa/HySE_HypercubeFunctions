"""

Functions used for wavelenght unmixing

"""

import numpy as np
from datetime import datetime
import matplotlib
from matplotlib import pyplot as plt
import inspect
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
from scipy.optimize import lsq_linear
matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"
from joblib import Parallel, delayed
import HySE.Masking


def MakeMixingMatrix_Flexible(Panel1_Wavelengths, Arduino_MixingMatrix_P1,
							  Panel2_Wavelengths, Arduino_MixingMatrix_P2, **kwargs):
	'''
	Computes a mixing matrix based on the Arduino matrices used during data recording.
	This version allows for a variable number of wavelengths to be combined in each image.

	Inputs:
		- Panel1_Wavelengths: List of wavelengths available in the first panel (e.g., REDs).
		- Arduino_MixingMatrix_P1: A list of lists. Each inner list contains the indices
		  of wavelengths from Panel1_Wavelengths to be combined for a single image.
		  Example: [[0, 1], [2, 3, 4], [5]] for 3 images.

		- Panel2_Wavelengths: List of wavelengths available in the second panel (e.g., BLUEs).
		- Arduino_MixingMatrix_P2: A list of lists, similar to P1 but for Panel2_Wavelengths.

		- kwargs:
			- Help: Print this information message
			- FromCalib = False: If true, use the single wavelength calibration hypercube
				to compute the mixing matrix.
			- Hypercube_WhiteCalib: Hypercube computed from the single wavelength calibration.
				Required if setting FromCalib to True.
			- UseMean = False: Sets the mean value of the calibration instead of the maximal value.
				Only used if FromCalib = True.
			- Plot=True: Plot the resulting mixing matrix.
			- Title : string
			- SaveFig = False: Save the plot to a file.
			- SavingPath = '': Path to save the figure.

	Outputs:
		- MixingMatrix: The computed mixing matrix. Dimensions are (total_images, total_unique_wavelengths).
	'''

	# --- Argument Handling ---
	Help = kwargs.get('Help', False)
	FromCalib = kwargs.get('FromCalib', False)
	UseMean = kwargs.get('UseMean', False)
	Plot = kwargs.get('Plot', True)
	Title = kwargs.get('Title', '')
	SaveFig = kwargs.get('SaveFig', False)
	SavingPath = kwargs.get('SavingPath', '')

	if Help:
		print(inspect.getdoc(MakeMixingMatrix_Flexible))
		return None

	if FromCalib:
		print('Computing mixing matrix from single wavelength calibration')
		Hypercube_WhiteCalib = kwargs.get('Hypercube_WhiteCalib')
		if Hypercube_WhiteCalib is None:
			print('MakeMixingMatrix error:')
			print('FromCalib has been set to True. Please provide Hypercube_WhiteCalib.')
			return None
	else:
		print('Computing binary mixing matrix')

	# --- Setup Matrix Dimensions ---
	# Combine all wavelengths and find the unique sorted list to define the matrix columns
	All_Wavelengths = list(Panel1_Wavelengths) + list(Panel2_Wavelengths)
	Wavelengths_sorted = np.unique(All_Wavelengths)
	N_unique_wavs = len(Wavelengths_sorted)

	# The number of rows is the total number of combined images from both panels
	N_images_P1 = len(Arduino_MixingMatrix_P1)
	N_images_P2 = len(Arduino_MixingMatrix_P2)
	N_total_images = N_images_P1 + N_images_P2

	# Initialize the potentially non-square mixing matrix
	MixingMatrix = np.zeros((N_total_images, N_unique_wavs))

	# --- Populate Mixing Matrix (Generalized Loop) ---
	# We group panel data to process them sequentially without repeating code
	panel_data = [
		(Panel1_Wavelengths, Arduino_MixingMatrix_P1),
		(Panel2_Wavelengths, Arduino_MixingMatrix_P2)
	]
	
	current_row_index = 0
	for panel_wavs, arduino_matrix in panel_data:
		# Iterate through each defined image combination (inner lists)
		for image_indices in arduino_matrix:
			# For each individual wavelength index in the current combination
			for wav_local_idx in image_indices:
				wav_nm = panel_wavs[wav_local_idx]
				# Find the column index in the final sorted list of all wavelengths
				wav_col_k = np.where(Wavelengths_sorted == wav_nm)[0][0]

				# Determine the amplitude (either 1 or from calibration)
				if FromCalib:
					if UseMean:
						wav_k_amp = np.nanmean(Hypercube_WhiteCalib[wav_col_k, :, :])
					else:
						wav_k_amp = np.nanmax(Hypercube_WhiteCalib[wav_col_k, :, :])
				else:
					wav_k_amp = 1
				
				# Assign the amplitude to the correct cell
				MixingMatrix[current_row_index, wav_col_k] = wav_k_amp
			
			current_row_index += 1 # Move to the next row for the next image

	# --- Compute Determinant (only if matrix is square) ---
	if MixingMatrix.shape[0] == MixingMatrix.shape[1]:
		Matrix_Det = np.linalg.det(MixingMatrix)
		if np.isclose(Matrix_Det, 0):
			print(f'WARNING! Matrix determinant is close to 0: {Matrix_Det}')
		else:
			print(f'Matrix determinant: {Matrix_Det}')
	else:
		print(f'Matrix is not square ({MixingMatrix.shape}), determinant not applicable.')

	# --- Plotting ---
	if Plot:
		N_rows, N_cols = MixingMatrix.shape
		row_ticks = np.arange(N_rows)
		col_ticks = np.arange(N_cols)
		row_labels = [f'im {i+1}' for i in row_ticks]
		col_labels = [f'{w:.0f}' for w in Wavelengths_sorted] # Format for clarity

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(max(5, N_cols*0.4), max(5, N_rows*0.4)))
		im = ax.imshow(MixingMatrix, cmap='magma', aspect='auto')
		
		ax.set_xticks(col_ticks)
		ax.set_yticks(row_ticks)
		ax.set_xticklabels(col_labels, rotation=90)
		ax.set_yticklabels(row_labels)

		ax.set_xlabel('Individual Wavelengths [nm]')
		ax.set_ylabel('Combined Image Index')
		
		if FromCalib:
			fig.colorbar(im, ax=ax, label='Weight')

		if FromCalib:
			title = 'Mixing Matrix - From Calibration' + (' (mean)' if UseMean else ' (max)')
		else:
			title = 'Mixing Matrix - Binary'
		title = title+'\n'+Title
		ax.set_title(title)
		
		plt.tight_layout()
		if SaveFig and SavingPath:
			plt.savefig(SavingPath)
			print(f"Figure saved to {SavingPath}")
		
		plt.show()

	return MixingMatrix


def MakeMixingMatrix(Wavelengths_unsorted, Arduino_MixingMatrix, **kwargs):
	'''
	Computes a mixing matrix, based on the Arduino matrix used during data recording
	If FromCalib is set to True, the mixing matrix is computed using the single wavelength
	sweep calibration. Otherwise, the function outputs a binary mixing matrix

	Inputs:
		- Wavelengths_unsorted: Unsorted wavelengths, as input in the NKT software 
		  list with 16 elements, panel 2, panel 4

		- Arduino_MixingMatrix: Arduino mixing matrix

		- kwargs:
			- Help: Print this information message
			- FromCalib = False: If true, use the single wavelength calibration hypercube
				to compute the mixing matrix
			- Hypercube_WhiteCalib: Hypercube computed from the single wavelength calibration
				Required if setting FromCalib to True
			- UseMean = False: Sets the mean value of the calibration instead of the maximal value
				Only used if FromCalib = True
			- Plot=True: Plot mixing matrix
			- SaveFig = False
			- SavingPath = ''

	Outputs:
		- MixingMatrix

	'''

	Help = kwargs.get('Help', False)
	FromCalib = kwargs.get('FromCalib', False)
	UseMean = kwargs.get('UseMean', False)
	Plot = kwargs.get('Plot', True)
	SaveFig = kwargs.get('SaveFig', False)
	SavingPath = kwargs.get('SavingPath', '')

	if Help:
		# print(info)
		print(inspect.getdoc(MakeMixingMatrix))
		return 0

	if FromCalib:
		print('Computing mixing matrix from single wavelength calibration')
		Hypercube_WhiteCalib = kwargs.get('Hypercube_WhiteCalib')
		if Hypercube_WhiteCalib is None:
			print(f'MakeMixingMatrix error:')
			print(f'FromCalib has been set to True. Please input the appropriate Hypercube_WhiteCalib')
			return None  # or raise an Exception
	else:
		print('Computing binary mixing matrix')


	Wavelengths_sorted = np.sort(Wavelengths_unsorted)
	Panel2_wavs = Wavelengths_unsorted[0:8]
	Panel4_wavs = Wavelengths_unsorted[8:]
	MixingMatrix = np.zeros((len(Wavelengths_sorted), len(Wavelengths_sorted)))
	(NN, Nmixed) = Arduino_MixingMatrix.shape
	for i in range(0,NN):
		## First profile (profile 2 - REDs)
		bins = Arduino_MixingMatrix[i,:]
		for j in range(0,len(bins)):
			wav_nm = Panel2_wavs[bins[j]]
			wav_k = np.where(Wavelengths_sorted==wav_nm)[0][0]
			if FromCalib:
				if UseMean:
					wav_k_amp = np.nanmean(Hypercube_WhiteCalib[wav_k,:,:])
				else:
					wav_k_amp = np.nanmax(Hypercube_WhiteCalib[wav_k,:,:])
			else:
				wav_k_amp = 1
			MixingMatrix[i,wav_k] = wav_k_amp

		## Second profile (profile 4 - BLUEs)
		for j in range(0,len(bins)):
			wav_nm = Panel4_wavs[bins[j]]
			wav_k = np.where(Wavelengths_sorted==wav_nm)[0][0]
			if FromCalib:
				if UseMean:
					wav_k_amp = np.nanmean(Hypercube_WhiteCalib[wav_k,:,:])
				else:
					wav_k_amp = np.nanmax(Hypercube_WhiteCalib[wav_k,:,:])
			else:
				wav_k_amp = 1
			MixingMatrix[i+8,wav_k] = wav_k_amp

	## Compute and print the matrix determinant
	Matrix_Det = np.linalg.det(MixingMatrix)
	if Matrix_Det==0:
		print(f'PROBLEM! Matrix determinant is 0')
	else:
		print(f'Matrix determinant: {Matrix_Det}')

	if Plot:
		xx = [i for i in range(0,16)]
		bin_labels = [f'im {i+1}' for i in range(0,16)]

		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
		ax.imshow(MixingMatrix, cmap='magma')
		ax.set_xticks(xx)
		ax.set_yticks(xx)
		ax.set_xticklabels(Wavelengths_sorted, rotation=90)
		ax.set_yticklabels(bin_labels)

		ax.set_xlabel(f'Individual Wavelengths [nm]')
		ax.set_ylabel(f'Combined Wavelengths [nm]')

		if FromCalib:
			if UseMean:
				ax.set_title(f'Mixing Matrix - From Calibration (mean)')
			else:
				ax.set_title(f'Mixing Matrix - From Calibration (max)')
		else:
			ax.set_title(f'Mixing Matrix - Binary')
		plt.tight_layout()
		if SaveFig:
			plt.savefig(SavingPath)

		plt.show()

	return MixingMatrix

def BlurWhiteCalibration(WhiteCalibration, Sigma):
	"""
	Function that applies Gaussian blurring to every frame in the white reference

	Inputs:
		- WhiteCalibration
		- Sigma (blurring)

	Outputs:
		- WhiteCalibrationBlurred

	"""
	(WW, YY, XX) = WhiteCalibration.shape
	WhiteCalibrationBlurred = np.zeros(WhiteCalibration.shape)
	for w in range(0,WW):
		cal_sub = WhiteCalibration[w,:,:]
		cal_sub_blurred = gaussian_filter(cal_sub, sigma=Sigma)
		WhiteCalibrationBlurred[w,:,:] = cal_sub_blurred
	return WhiteCalibrationBlurred

def SubtractDark(WhiteCalibration, Dark_g):
	"""
	Function that subtacts dark from every frame in hypecube

	Inputs:
		- WhiteCalibration
		- Dark_g

	Outputs:
		- WhiteCalibration_D

	"""
	(WW, YY, XX) = WhiteCalibration.shape
	WhiteCalibration_D = np.zeros(WhiteCalibration.shape)
	for w in range(0,WW):
		cal_sub = WhiteCalibration[w,:,:]
		cal_sub_d = np.subtract(cal_sub, Dark_g)
		WhiteCalibration_D[w,:,:] = cal_sub_d
	return WhiteCalibration_D



def NormaliseMixedHypercube(MixedHypercube, **kwargs):
	'''
	Normalises the raw mixed hypercube.
	Dark subtraction and/or white normalisation

	Input:
		- MixedHypercube

		- kwargs:
			- Help
			- Dark = array shape (YY, XX)
			- WhiteCalibration = array shape (Nwavelengths, YY, XX)
			- Sigma = 20. Integer, indicates how much blurring to apply 
				for the dark and White calibration arrays
			- SpectralNormalisation = False. If True, the white reference will be used to normalise the spectrum
				instead of both spectrally and spatially.
			- vmax: float. For plotting range
			- vmin: float. For plotting range
			- SavePlot = False
			- SavingFigure = '' string.
			- Plot = True
			- IndivFrames = False : Whether or not individual frames have been kept for each instead of being averaged


	Outputs:
		- Normalised Hypercube

		- Mask
			Removes black corners from the Olympus display. Estimated from the white calibration if provided,
			otherwise it uses the data itself to estimate the mask

	'''

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(NormaliseMixedHypercube))
		return 0,0

	Sigma = kwargs.get('Sigma', 20)
	Dark = kwargs.get('Dark')
	if Dark is not None:
		HypercubeShape = MixedHypercube.shape
		X = HypercubeShape[-1]
		Y = HypercubeShape[-2]
		(Yd, Xd) = Dark.shape
		print(f'Checking if dark needs to be cropped: Hypercube X={X}, Y={Y}, Dark Xd={Xd}, Yd={Yd}')
		if ((Y!=Yd) or (X!=Xd)):
			print(f'Data ({MixedHypercube.shape}) is cropped. Cropping Dark ({Dark.shape})')
			ToCropX = Xd-X
			ToCropY = Yd-Y
			if ToCropX!=ToCropY:
				print(f'Cropping doesn\'t seem to be the same along both axes. X: {ToCropX}, Y: {ToCropY}')

			c = int(ToCropX/2)
			Dark_ = Dark[c:-c, (c+1):(-c)]
			print(f'Dark new size: {Dark_.shape}')
		else: Dark_ = Dark


		Dark_g = gaussian_filter(Dark_, sigma=Sigma)
		print(f'Dark subtraction. Avg val = {np.average(Dark_):.2f}, after blurring: {np.average(Dark_g):.2f}')
		## Check if sizes match (if data is cropped)
		

	SpectralNormalisation = kwargs.get('SpectralNormalisation', False)
	IndivFrames = kwargs.get('IndivFrames', False)

	WhiteCalibration = kwargs.get('WhiteCalibration')
	if WhiteCalibration is not None:
		print(f'White Normalising')
		if SpectralNormalisation:
			print(f'Only spectral normalisation')
		# Wavelengths_list = kwargs.get('Wavelengths_list')
		# if Wavelengths_list is None:
		# 	print(f'Please indicate Wavelengths_list (unsorted) when requesting WhiteCalibration')
		# 	return 0
		if Dark is not None:
			WhiteCalibration_ = BlurWhiteCalibration(SubtractDark(WhiteCalibration, Dark_g), Sigma)
			print(f'Subtracting Dark and blurring White Reference with sigma = {Sigma}')
			print(f'WC-Dark: {np.average(np.average(WhiteCalibration_, axis=1), axis=1)}')
		else:
			WhiteCalibration_ = BlurWhiteCalibration(WhiteCalibration, Sigma)
			print(f'Blurring White Reference with sigma = {Sigma}')
			print(f'WC: {np.average(np.average(WhiteCalibration_, axis=1), axis=1)}')

	if (SpectralNormalisation and WhiteCalibration is None):
				print(f'Please input WhiteCalibration when requesting SpectralNormalisation')
				return 0

	Plot = kwargs.get('Plot', True)
	SaveFigure = kwargs.get('SaveFigure', True)
	SavingPath = kwargs.get('SavingPath', '')
	vmax = kwargs.get('vmax', 5)
	vmin = kwargs.get('vmin', 0)
	if (Dark is None) and (WhiteCalibration is None):
		print(f'No normalisation')
		Plot=False


	if len(MixedHypercube.shape)==3:
		MixedHypercube_ = np.array([[MixedHypercube]])
	elif len(MixedHypercube.shape)==4:
		MixedHypercube_ = np.array([MixedHypercube])
	elif len(MixedHypercube.shape)==5:
		MixedHypercube_ = MixedHypercube
	else:
		print(f'Shape of the Mixed hypercube not supported: len(Hypercube.shape) = {len(MixedHypercube.shape)} != [3,4,5]')

	if WhiteCalibration is not None:
		Mask = HySE.Masking.GetStandardMask(WhiteCalibration_, threshold=1)
	else:
		print(f'White Calibration not provided. Estimating mask from data itself.')
		Mask = HySE.Masking.GetStandardMask(MixedHypercube_[0], threshold=1)

	MixedHypercube_N = np.zeros(MixedHypercube_.shape)
	(SS, WW, FF, YY, XX) = MixedHypercube_.shape
	MixedHypercube_N = np.zeros(MixedHypercube_.shape)
	for s in range(0,SS): 
		for w in range(0,WW):
			for f in range(0,FF):
				frame = MixedHypercube_[s,w,f,:,:]
				if Dark is not None:
					frameD = np.subtract(frame, Dark_g)
				else:
					frameD = frame

				if WhiteCalibration is None:
					frameN = frameD
				else:
					whiteframe = WhiteCalibration_[w,:,:]
					if SpectralNormalisation:
						whiteval = np.average(whiteframe)
						frameN = frameD/whiteval
					else:
						frameN = np.divide(frameD, whiteframe, out=np.zeros_like(frameD), where=whiteframe!=0)
				MixedHypercube_N[s,w,f,:,:] = frameN

	if len(MixedHypercube.shape)==3: ## Just wavelengths, Y, X
		MixedHypercube_N_ = MixedHypercube_N[0,0,:,:,:]
	elif len(MixedHypercube.shape)==4: 
		if IndivFrames: ## Single sweep
			print(f'POTENTIAL ERROR: CHECK THAT THE HYPERCUBE OUTPUT DIMENSIONS ARE CORRECT')
			print(f'   MixedHypercube_N.shape: {MixedHypercube_N.shape}')
			print(f'   Currently reshaped as [0,:,:,:,:]')
			MixedHypercube_N_ = MixedHypercube_N[0,:,:,:,:]
		else: ## One sweeo but multiple frames
			print(f'POTENTIAL ERROR: CHECK THAT THE HYPERCUBE OUTPUT DIMENSIONS ARE CORRECT')
			print(f'   MixedHypercube_N.shape: {MixedHypercube_N.shape}')
			print(f'   Currently reshaped as [0,:,:,:,:]')
			MixedHypercube_N_ = MixedHypercube_N[:,0,:,:,:]
	elif len(MixedHypercube.shape)==5:
		MixedHypercube_N_ = MixedHypercube_N


	if Plot:
		if len(MixedHypercube.shape)==3: ## Just wavelengths, Y, X
			HypercubeToPlot = MixedHypercube_N_
		elif len(MixedHypercube.shape)==4: 
			if IndivFrames: 
				HypercubeToPlot = MixedHypercube_N_[:,0,:,:]
			else:
				HypercubeToPlot = MixedHypercube_N_[0,:,:,:]
		elif len(MixedHypercube.shape)==5:
			HypercubeToPlot =MixedHypercube_N_[0,:,0,:,:]

		nn = 0
		Mavg = np.average(HypercubeToPlot)
		Mstd = np.std(HypercubeToPlot)
		fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
		print(f'Plotting shapes: HypercubeToPlot -> {HypercubeToPlot.shape}, Mask -> {Mask.shape} ')
		for j in range(0,4):
			for i in range(0,4):
				if nn<17:
					print(HypercubeToPlot[nn,:,:].shape)
					print(Mask.shape)
					# ToPlot_sub = np.ma.array(HypercubeToPlot[nn,:,:], mask=Mask[nn,:,:])à
					ToPlot_sub = HypercubeToPlot[nn,:,:]
					ax[j,i].imshow(ToPlot_sub, cmap='magma',vmin=vmin, vmax=vmax)
					ax[j,i].set_title(f'im {nn}')
					ax[j,i].set_xticks([])
					ax[j,i].set_yticks([])
					nn = nn+1
				else:
					ax[j,i].set_xticks([])
					ax[j,i].set_yticks([])
		plt.tight_layout()
		if SaveFigure:
			plt.savefig(f'{SavingPath}Normalised_Hypercube.png')
		plt.show()

	return MixedHypercube_N_, Mask


def UnmixData(MixedHypercube, MixingMatrix, **kwargs):
	'''
	Computes the unmixing of the raw hypercube according to a mixing matrix

	Input:
		- MixedHypercube
		- Mixing Matrix

		- kwargs:
			- Help
			- Average = True. If the input mixed hypercube containes more than one shape,
				the function can average the unmixed arrays or leave then individually

	Output:
		- Unmixed hypercube


	'''

	Help = kwargs.get('Help', False)
	Average = kwargs.get('Average', True)
	if Help:
		# print(info)
		print(inspect.getdoc(UnmixData))
		return 0

	## Adjust shape of the Hypercube for consistency
	if len(MixedHypercube.shape)>3:
		MixedHypercube_ = MixedHypercube
	else:
		MixedHypercube_ = np.array([MixedHypercube])
		
	UnmixedHypercube = []
	SS, WW, YY, XX = MixedHypercube_.shape
	for s in range(0,SS):
		hypercube_sub = MixedHypercube_[s,:,:,:]
		NN, YY, XX = hypercube_sub.shape
		ObservedMatrix = MakeObservedMatrix(hypercube_sub)
		SolvedMatrix_flat = np.linalg.lstsq(MixingMatrix, ObservedMatrix, rcond=-1)[0]
		SolvedMatrix = []
		for n in range(0,NN):
			matrix_sub = SolvedMatrix_flat[n,:].reshape(YY,XX)
			SolvedMatrix.append(matrix_sub)
		UnmixedHypercube.append(SolvedMatrix)
	
	UnmixedHypercube = np.array(UnmixedHypercube)
	if Average:
		UnmixedHypercube = np.average(UnmixedHypercube,axis=0)
	return UnmixedHypercube

		
		
def MakeObservedMatrix(Hypercube):
	"""
	Make observed matrix (ravel) from the hypercube to feed in the unmixing algorithm

	Inputs:
		- Hypercube

	Outptus:
		- ObservedMatrix


	"""
	N, YY, XX = Hypercube.shape
	Matrix = []
	for n in range(0,N):
		data_sub = Hypercube[n,:,:]
		Matrix.append(data_sub.ravel())
	return np.array(Matrix)



def UnmixDataNNLS(MixedHypercube, MixingMatrix, intensity_thresh=1e-2, std_thresh=1e-3,
				  max_iter=5000, parallel=True, **kwargs):
	"""
	Unmix a hypercube using non-negative least squares with noise filtering
	and rejection of pixels with negative or invalid total intensity.

	Parameters:
		- MixedHypercube : array, shape (S, N, Y, X) or (N, Y, X)
		- MixingMatrix : array, shape (num_images, num_wavelengths)
		- intensity_thresh = 1e-2 : float, reject very dim pixels
		- std_thresh = 1e-3 : float, reject flat/noisy pixels
		- max_iter = 5000 : int, maximum iterations for solver
		- parallel = False : bool, enable parallel processing
		- kwargs
			- Help: Show this help message
			- Average = True. If the input mixed hypercube containes more than one shape,
				the function can average the unmixed arrays or leave then individually

	Output:
		- Unmixed Hypercube

	"""
	Help = kwargs.get('Help', False)
	Average = kwargs.get('Average', True)
	if Help:
		print(inspect.getdoc(UnmixDataNNLS))
		return 0

	if len(MixedHypercube.shape) > 3:
		MixedHypercube_ = MixedHypercube
	else:
		MixedHypercube_ = np.array([MixedHypercube])

	UnmixedHypercube = []
	SS, WW, YY, XX = MixedHypercube_.shape

	for s in range(SS):
		hypercube_sub = MixedHypercube_[s]
		NN, YY, XX = hypercube_sub.shape
		ObservedMatrix = HySE.MakeObservedMatrix(hypercube_sub).T  # shape: (pixels, n_meas)
		num_pixels = ObservedMatrix.shape[0]
		num_waves = MixingMatrix.shape[1]
		SolvedMatrix_flat = np.zeros((num_waves, num_pixels))

		def solve_single(i):
			pixel = ObservedMatrix[i, :]
			if (np.sum(pixel) < intensity_thresh or
				np.std(pixel) < std_thresh or
				np.sum(pixel) < 0):
				return np.zeros(num_waves)
			res = lsq_linear(MixingMatrix, pixel, bounds=(0, np.inf), max_iter=max_iter, method='trf')
			return res.x if res.success else np.zeros(num_waves)

		if parallel:
			# from joblib import Parallel, delayed
			results = Parallel(n_jobs=-1)(delayed(solve_single)(i) for i in range(num_pixels))
			SolvedMatrix_flat = np.array(results).T
		else:
			for i in range(num_pixels):
				SolvedMatrix_flat[:, i] = solve_single(i)

		SolvedMatrix = [SolvedMatrix_flat[n, :].reshape(YY, XX) for n in range(num_waves)]
		UnmixedHypercube.append(SolvedMatrix)

	UnmixedHypercube = np.array(UnmixedHypercube)
	if Average:
		UnmixedHypercube = np.mean(UnmixedHypercube, axis=0)
	return UnmixedHypercube



def UnmixDataSmoothNNLS(MixedHypercube, MixingMatrix, CombinedMask=None, lambda_smooth=0.1,
						intensity_thresh=1e-2, std_thresh=1e-3,
						max_iter=5000, parallel=True, **kwargs):
	"""
	Unmix hypercube using non-negative least squares with Tikhonov regularization
	promoting smoothness across wavelengths.

	Parameters:
		- MixedHypercube : array, shape (S, N, Y, X) or (N, Y, X)
		- MixingMatrix : array, shape (num_images, num_wavelengths)
		- CombinedMask : 2D array (Y, X), optional. Boolean mask where True
						 indicates invalid pixels to skip. # <-- NEW
		- lambda_smooth = 0.1 : regularization strength for spectral smoothness
		- intensity_thresh = 1e-2: intensity threshold for skipping noisy pixels (aim for ~ 1e-2*max(Hypercube))
		- std_thresh = 1e-3 : standard deviation threshold for skipping noisy pixels
		- max_iter = 5000 : max iterations for NNLS solver
		- parallel = True: use parallel joblib processing
		- kwargs
			- Help: Show this help message
			- Average = True. If the input mixed hypercube containes more than one shape,
				the function can average the unmixed arrays or leave then individually

	Output:
		-  Unmixed Hypercube
	"""
	Help = kwargs.get('Help', False)
	Average = kwargs.get('Average', True)
	if Help:
		print(inspect.getdoc(UnmixDataSmoothNNLS))
		return 0

	if len(MixedHypercube.shape) > 3:
		MixedHypercube_ = MixedHypercube
	else:
		MixedHypercube_ = np.array([MixedHypercube])

	UnmixedHypercube = []
	SS, WW, YY, XX = MixedHypercube_.shape
	num_meas, num_waves = MixingMatrix.shape

	# Construct second-difference matrix L (smoothness penalty)
	L = -2 * np.eye(num_waves) + np.eye(num_waves, k=1) + np.eye(num_waves, k=-1)
	A_reg_base = np.vstack([MixingMatrix, np.sqrt(lambda_smooth) * L])
	zeros_rhs = np.zeros((num_waves,))

	for s in range(SS):
		hypercube_sub = MixedHypercube_[s]
		NN, YY, XX = hypercube_sub.shape
		ObservedMatrix = HySE.MakeObservedMatrix(hypercube_sub).T # shape: (pixels, n_meas)
		num_pixels = ObservedMatrix.shape[0]
		SolvedMatrix_flat = np.zeros((num_waves, num_pixels))

		mask_flat = None
		if CombinedMask is not None:
			if CombinedMask.shape != (YY, XX):
				# Add a check to ensure mask dimensions are correct
				raise ValueError(f"CombinedMask shape {CombinedMask.shape} does not match hypercube spatial dimensions ({YY}, {XX})")
			mask_flat = CombinedMask.flatten()

		# def solve_single(i):
		# 	# Check the provided mask first. If True, the pixel is invalid.
		# 	if mask_flat is not None and mask_flat[i]: 
		# 		return np.zeros(num_waves)

		# 	# Original checks for intensity and standard deviation
		# 	pixel = ObservedMatrix[i, :]
		# 	if np.sum(pixel) < 0 or np.sum(pixel) < intensity_thresh or np.std(pixel) < std_thresh:
		# 		return np.zeros(num_waves)

		# 	b_reg = np.concatenate([pixel, zeros_rhs])
		# 	res = lsq_linear(A_reg_base, b_reg, bounds=(0, np.inf), max_iter=max_iter, method='trf')
		# 	return res.x if res.success else np.zeros(num_waves)

		def solve_single(i):
			# Check the provided mask first. If True, the pixel is invalid.
			if mask_flat is not None and mask_flat[i]: 
				return np.zeros(num_waves)

			# Original checks for intensity and standard deviation
			pixel = ObservedMatrix[i, :]
			
			# If any value in the pixel's spectrum is NaN, skip it.
			if np.isnan(pixel).any(): # <-- NEW NAN CHECK
				return np.zeros(num_waves)

			if np.sum(pixel) < 0 or np.sum(pixel) < intensity_thresh or np.std(pixel) < std_thresh:
				return np.zeros(num_waves)

			b_reg = np.concatenate([pixel, zeros_rhs])
			res = lsq_linear(A_reg_base, b_reg, bounds=(0, np.inf), max_iter=max_iter, method='trf')
			return res.x if res.success else np.zeros(num_waves)

		if parallel:
			# from joblib import Parallel, delayed # (Already imported above)
			results = Parallel(n_jobs=-1)(delayed(solve_single)(i) for i in range(num_pixels))
			SolvedMatrix_flat = np.array(results).T
		else:
			for i in range(num_pixels):
				SolvedMatrix_flat[:, i] = solve_single(i) # The logic inside solve_single handles the skip

		SolvedMatrix = [SolvedMatrix_flat[n, :].reshape(YY, XX) for n in range(num_waves)]
		UnmixedHypercube.append(SolvedMatrix)

	UnmixedHypercube = np.array(UnmixedHypercube)
	if Average:
		UnmixedHypercube = np.mean(UnmixedHypercube, axis=0)
	return UnmixedHypercube



	

# def UnmixDataSmoothNNLS(MixedHypercube, MixingMatrix, lambda_smooth=0.1,
# 						intensity_thresh=1e-2, std_thresh=1e-3,
# 						max_iter=5000, parallel=True, **kwargs):
# 	"""
# 	Unmix hypercube using non-negative least squares with Tikhonov regularization
# 	promoting smoothness across wavelengths.

# 	Parameters:
# 		- MixedHypercube : array, shape (S, N, Y, X) or (N, Y, X)
# 		- MixingMatrix : array, shape (num_images, num_wavelengths)
# 		- lambda_smooth = 0.1 : regularization strength for spectral smoothness
# 		- intensity_thresh = 1e-2: intensity threshold for skipping noisy pixels (aim for ~ 1e-2*max(Hypercube))
# 		- std_thresh = 1e-3 : standard deviation threshold for skipping noisy pixels
# 		- max_iter = 5000 : max iterations for NNLS solver
# 		- parallel = True: use parallel joblib processing
# 		- kwargs
# 			- Help: Show this help message
# 			- Average = True. If the input mixed hypercube containes more than one shape,
# 				the function can average the unmixed arrays or leave then individually

# 	Output:
# 		-  Unmixed Hypercube
# 	"""
# 	Help = kwargs.get('Help', False)
# 	Average = kwargs.get('Average', True)
# 	if Help:
# 		print(inspect.getdoc(UnmixDataSmoothNNLS))
# 		return 0

# 	if len(MixedHypercube.shape) > 3:
# 		MixedHypercube_ = MixedHypercube
# 	else:
# 		MixedHypercube_ = np.array([MixedHypercube])

# 	UnmixedHypercube = []
# 	SS, WW, YY, XX = MixedHypercube_.shape
# 	num_meas, num_waves = MixingMatrix.shape

# 	# Construct second-difference matrix L (smoothness penalty)
# 	L = -2 * np.eye(num_waves) + np.eye(num_waves, k=1) + np.eye(num_waves, k=-1)
# 	A_reg_base = np.vstack([MixingMatrix, np.sqrt(lambda_smooth) * L])
# 	zeros_rhs = np.zeros((num_waves,))

# 	for s in range(SS):
# 		hypercube_sub = MixedHypercube_[s]
# 		NN, YY, XX = hypercube_sub.shape
# 		ObservedMatrix = HySE.MakeObservedMatrix(hypercube_sub).T  # shape: (pixels, n_meas)
# 		num_pixels = ObservedMatrix.shape[0]
# 		SolvedMatrix_flat = np.zeros((num_waves, num_pixels))

# 		def solve_single(i):
# 			pixel = ObservedMatrix[i, :]
# 			if np.sum(pixel) < 0 or np.sum(pixel) < intensity_thresh or np.std(pixel) < std_thresh:
# 				return np.zeros(num_waves)
# 			b_reg = np.concatenate([pixel, zeros_rhs])
# 			res = lsq_linear(A_reg_base, b_reg, bounds=(0, np.inf), max_iter=max_iter, method='trf')
# 			return res.x if res.success else np.zeros(num_waves)

# 		if parallel:
# 			# from joblib import Parallel, delayed
# 			results = Parallel(n_jobs=-1)(delayed(solve_single)(i) for i in range(num_pixels))
# 			SolvedMatrix_flat = np.array(results).T
# 		else:
# 			for i in range(num_pixels):
# 				SolvedMatrix_flat[:, i] = solve_single(i)

# 		SolvedMatrix = [SolvedMatrix_flat[n, :].reshape(YY, XX) for n in range(num_waves)]
# 		UnmixedHypercube.append(SolvedMatrix)

# 	UnmixedHypercube = np.array(UnmixedHypercube)
# 	if Average:
# 		UnmixedHypercube = np.mean(UnmixedHypercube, axis=0)
# 	return UnmixedHypercube




def UnmixDataSmoothNNLSPrior(MixedHypercube, MixingMatrix, prior_spectrum,
							lambda_smooth=0.1, lambda_prior=0.1,
							intensity_thresh=1e-2, std_thresh=1e-3,
							max_iter=5000, parallel=True, **kwargs):
	"""
	Unmix hypercube using NNLS with two regularizers:
	1. Tikhonov regularization promoting spectral smoothness.
	2. A penalty term that pulls the solution towards a known prior_spectrum.

	Parameters:
		- MixedHypercube : array, shape (S, N, Y, X) or (N, Y, X)
		- MixingMatrix : array, shape (num_images, num_wavelengths)
		- prior_spectrum : array, shape (num_wavelengths,). Your calibration/reference.
		- lambda_smooth = 0.1 : Regularization for spectral smoothness.
		- lambda_prior = 0.1 : Regularization for closeness to the prior.
		... (rest of the parameters are the same) ...

	Output:
		- Unmixed Hypercube
	"""
	Help = kwargs.get('Help', False)
	Average = kwargs.get('Average', True)
	if Help:
		print(inspect.getdoc(UnmixDataSmoothPriorNNLS))
		return 0

	if len(MixedHypercube.shape) > 3:
		MixedHypercube_ = MixedHypercube
	else:
		MixedHypercube_ = np.array([MixedHypercube])

	UnmixedHypercube = []
	SS, WW, YY, XX = MixedHypercube_.shape
	num_meas, num_waves = MixingMatrix.shape

	# --- NEW: Check shape of the prior spectrum ---
	if prior_spectrum.shape[0] != num_waves:
		raise ValueError(f"Shape of prior_spectrum ({prior_spectrum.shape[0]}) does not match "
						 f"number of wavelengths in MixingMatrix ({num_waves}).")

	# --- MODIFIED: Construct the "triple-decker" problem matrix ---
	# 1. Smoothness penalty matrix (L)
	L = -2 * np.eye(num_waves) + np.eye(num_waves, k=1) + np.eye(num_waves, k=-1)
	
	# 2. Prior penalty matrix (Identity matrix, I)
	I = np.eye(num_waves)
	
	# 3. Stack all three matrices
	A_reg_base = np.vstack([
		MixingMatrix,
		np.sqrt(lambda_smooth) * L,
		np.sqrt(lambda_prior) * I
	])

	# Prepare the constant parts of the right-hand side (b) vector
	zeros_rhs_smooth = np.zeros((num_waves,))
	# --- NEW: The target for the prior penalty ---
	prior_rhs = np.sqrt(lambda_prior) * prior_spectrum

	for s in range(SS):
		hypercube_sub = MixedHypercube_[s]
		ObservedMatrix = HySE.MakeObservedMatrix(hypercube_sub).T
		num_pixels = ObservedMatrix.shape[0]
		SolvedMatrix_flat = np.zeros((num_waves, num_pixels))

		def solve_single(i):
			pixel = ObservedMatrix[i, :]
			if np.sum(pixel) < 0 or np.sum(pixel) < intensity_thresh or np.std(pixel) < std_thresh:
				return np.zeros(num_waves)
			
			# --- MODIFIED: Construct the augmented b vector for each pixel ---
			b_reg = np.concatenate([pixel, zeros_rhs_smooth, prior_rhs])
			
			res = lsq_linear(A_reg_base, b_reg, bounds=(0, np.inf), max_iter=max_iter, method='trf')
			return res.x if res.success else np.zeros(num_waves)

		if parallel:
			# from joblib import Parallel, delayed
			results = Parallel(n_jobs=-1)(delayed(solve_single)(i) for i in range(num_pixels))
			SolvedMatrix_flat = np.array(results).T
		else:
			for i in range(num_pixels):
				SolvedMatrix_flat[:, i] = solve_single(i)

		SolvedMatrix = [SolvedMatrix_flat[n, :].reshape(YY, XX) for n in range(num_waves)]
		UnmixedHypercube.append(SolvedMatrix)

	UnmixedHypercube = np.array(UnmixedHypercube)
	if Average:
		UnmixedHypercube = np.mean(UnmixedHypercube, axis=0)
	return UnmixedHypercube


def omit_frames(arr, indices):
	if arr.shape[0] != 16:
		raise ValueError(f"Expected first dimension = 16, got {arr.shape[0]}")
	keep_indices = list(indices)+list([o+8 for o in indices])
	print(f'Original indices: {indices}')
	print(f'Keeping indices: {keep_indices}')
	return arr[keep_indices]


def combine_hypercubes(hypercube1, chypercube2, wavelengths1, wavelengths2):
	"""
	Combine two hyperspectral sub-cubes into one cube sorted by wavelength.

	Inputs:
		- hypercube1: first sub hypercube to combine (size N, Y, X)
		- hypercube2: second sub hypercube to combine (size M, Y, X)
		- wavelengths1: wavelengths used for hypercube1 (size/length N)
		- wavelengths2: wavelengths used for hypercube2 (size/length M)

	Outputs:
		- ccombined_hypercube: size (N+M, Y, X)
		- combined_wavelengths: size/length N+M
	"""

	# Stack cubes and wavelengths
	all_hypercubes = np.concatenate([hypercube1, hypercube2], axis=0)
	print(all_hypercubes.shape)
	all_wavelengths = np.array(list(wavelengths1) + list(wavelengths2))

	# Sort indices by wavelength
	sort_idx = np.argsort(all_wavelengths)

	# Reorder
	combined_hypercube = all_hypercubes[sort_idx]
	combined_wavelengths = all_wavelengths[sort_idx]

	return combined_hypercube, combined_wavelengths
		