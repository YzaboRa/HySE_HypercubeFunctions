"""
Created on 18th November 2025 by Katie-Lou White for the HySE analysis pipeline
Designed to normalise each frame by the reference channel.

This should be done after dark normalisation, prior to co-registration and unmixing
"""

import numpy as np
import matplotlib.pyplot as plt
import HySE


def GetChannelInterplay(file,dark_means,buffer=10,HighPixVal=200,LowPixVal=50,return_mask=False):
    print("Returns a correction matrix (M) to convert videos (v) to intensities (i) as Cv=i")
    if return_mask:
        print("Also returning the mask of saturated and very dim pixels during white referencing")
    """Hardcoded Data from Arduino Code"""
    # Sweep Exposure Times
    BSweep = [7000,6500,6000,5500,5000,4500,4000,3500,3000,2500,2000,1500,1000,500,0]
    GSweep = [6500,6000,5500,5000,4500,4000,3500,3000,2500,2000,1500,1000,500,0]
    RSweep = [1600,1500,1400,1300,1200,1100,1000,900,800,700,600,500,400,300,200,100,0]
    BGRTest = np.array([[7000,0,6500,500,6000,1000,5500,1500,5000,2000,4500,2500,3500,3000],
                        [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500],
                        [1600,1500,1400,1300,1200,1100,1000,900,800,700,600,500,400,300]])
    n_B,n_G,n_R,n_mix = len(BSweep), len(GSweep), len(RSweep),len(BGRTest[0]) # Number of exposure times for each sweep set

    # Peak Durations
    t_on = 10 # Number of frames with light on for each exposure time
    t_off = 2 # Number of frames with light off in between
    t_single = 3*(t_on+t_off) # Time for a single illumination peak & trough
    t_repeat = (n_B+n_G+n_R+n_mix+1)*t_single # Number of frames in a single cycle

    # Extract dark, and trace of whole video without masking
    trace = HySE.Import.ImportData(file,RGB=True, Trace=True,TrimSat=False)
    fig1,ax1=plt.subplots()
    for i, col in enumerate(['b', 'g', 'r']):
        ax1.plot(trace[:,i],color=col,alpha=0.2)

    # Find starts of each sweep
    # Plotting limits
    y_max = np.nanmax(trace)
    x_width = 2*t_single
    starts = np.zeros((3,2))
    # Find all the single-color exposure sweeps
    for i, col in enumerate(['b', 'g', 'r']):
        for rpt in [0,1]:
            x_start = ([n_R+n_G,n_R,0][i]*t_single)+t_repeat*rpt
            # ax[rpt,2-i].plot(x_start+np.arange(0,x_width),trace[x_start:x_start+x_width,i],color=col,alpha=0.2)

            # Find peak at start
            peak = np.argmax(np.diff(trace[x_start:x_start+x_width,i]))
            ax1.vlines(x_start+peak,0,y_max,color='k',linestyle=['-','--'][rpt])
            starts[i, rpt] = peak+x_start

    # Find start of mixed sweep
    x_start = (n_R+n_G+n_B)*t_single
    peak = np.argmax(np.diff(trace[x_start:x_start + x_width, 2]))
    mix_start = x_start + peak
    ax1.vlines(mix_start, 0, y_max, color='gray', linestyle='dotted')

    # Mask out saturated pixels and dimly illuminated areas, then recalculate trace without saturation
    i=1 # Only look at brightest green frame, as green is always much brighter than blue and red.
    img = HySE.Import.ImportData(file, RGB=True, Trace=False,TrimSat=False,Coords=[int(starts[i,rpt]+buffer),int(starts[i,rpt]+t_single-buffer)]).mean(axis=0)

    fig2,ax2=plt.subplots(1,2,sharex=True,sharey=True)
    ax2[0].imshow(img/255)
    # Find saturated values and remove them from the image by setting them to NaNs
    mask = 1-(img.max(axis=2)<HighPixVal)*(img.max(axis=2)>LowPixVal)
    ax2[1].imshow(mask)

    trace = HySE.Import.ImportData(file, Mask=mask.T,RGB=True, Trace=True,TrimSat=False)
    for i, col in enumerate(['b', 'g', 'r']):
        ax1.plot(trace[:,i],color=col,alpha=1)

    # Find each peak and its intensity for each colour channel sweep
    fig,ax=plt.subplots(1,3,sharex=True,sharey=True)
    fig.suptitle("Colour Channel Interplay Signals")
    all_values = []
    darks = []
    for rpt in [0, 1]:
        for i, col1 in enumerate(['b', 'g', 'r']):
            # Get number of exposure times and start of trace
            values, startset, n = [], starts[i], [n_B,n_G,n_R][i]
            # Loop through peaks, guessing and refining start time

            for n_j in range(n):
                start = int(startset[rpt])+n_j*t_single

                # # Possible Improvement: Find actual time for each peak rather than just relying on the buffer
                # subtrace = trace[start-t_off:start+t_on]
                # peak = np.argmax(subtrace)+start # Maybe incorporate later to refine position
                means = np.mean(trace[start+buffer:start+t_single-buffer],axis=0)
                values.append(means)

                # Plot
                for j, col2 in enumerate(['b', 'g', 'r']):
                    ax[i].scatter([BSweep,GSweep,RSweep][i][n_j],means[j],color=col2,marker='o',alpha=0.5)

            if len(all_values)==0:
                all_values=np.array(values)
            else:
                all_values=np.concatenate((all_values,values))

    # Generate list of exposure times for each case
    for i, exps_i in enumerate([BSweep,GSweep,RSweep]):
        exps = np.zeros((len(exps_i),3))
        exps[:,i] = exps_i
        if i==0:
            all_exps = exps
        else:
            all_exps = np.concatenate((all_exps,exps))
    all_exps = np.concatenate((all_exps,all_exps))


    # Calculate gradients and correction matrix
    gradients = np.linalg.lstsq(all_exps,all_values-dark_means,rcond=None)[0]
    correction_matrix = np.linalg.inv(gradients)

    trace_correct = np.matmul(trace-dark_means,correction_matrix)

    x = np.linspace(0,5000,10)
    for i, col1 in enumerate(['b', 'g', 'r']):
        for j, col2 in enumerate(['b', 'g', 'r']):
            ax[i].plot(x,gradients[i,j]*x+dark_means[j],color=col2,alpha=0.2)

    # Demonstrate Trace with Channel Interplay Correction Applied
    fig,ax=plt.subplots(2,sharex=True)
    fig.suptitle("Channel Interplay Mean Intensities")
    for i, col in enumerate(['b', 'g', 'r']):
        ax[0].plot(trace[:,i],color=col,alpha=0.8),         ax[0].set_title("Raw Data")
        ax[1].plot(trace_correct[:,i],color=col,alpha=0.8), ax[1].set_title("Corrected Data")

    if return_mask:
        return correction_matrix,mask
    else:
        return correction_matrix

