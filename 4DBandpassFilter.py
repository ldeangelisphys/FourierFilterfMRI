# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:20:16 2019

@author: deangelis
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.signal import tukey



def make_mask(freq,bounds,window_type, alpha = 0.1):
    
    mask = np.zeros(len(freq))
    toselect = (freq > bounds[0]) * (freq < bounds[1])
    
    if window_type == 'Hanning':
        window_f = np.hanning
    elif window_type == 'Blackman':
        window_f = np.blackman
    elif window_type == 'Tukey':
        window_f = lambda x: tukey(x, alpha = alpha)
    else:
        window_f = lambda x: np.ones(x)
    
    window = window_f(np.sum(toselect))
    mask[toselect] += window
    
    return mask

def init_masks():
    
    
    masks = {}
    known_windows = ['Hanning','Blackman','Tukey','Rectangular']
    for window in known_windows:
        masks[window] = make_mask(DATAFTFREQ,freqbound,window)
    
    return masks



def plot_filter(XDATA,DATA,DATAFILTER, masks, wm, fout, in_fourier = True):

    
    fig,axs = plt.subplots(nrows = 2, figsize  = (15,5), sharex = True)
    
    xc,yc,zc,tc = np.array( np.array(DATA.shape) / 2 , dtype = int)
    
    axs[0].plot(XDATA,np.abs(DATA[xc,yc,zc,:]))
    axs[1].plot(XDATA,np.abs(DATAFILTER[xc,yc,zc,:]))
    
    maxft = np.max(np.abs(DATA[xc,yc,zc,:]))
    
 
    if in_fourier:
        axs[0].fill_between(XDATA,masks[wm] * maxft,alpha = 0.3, color = 'C2', label = '{} mask'.format(wm) )
        axs[1].set_xlabel('Frequency (Hz)')   
        axs[0].legend()

    else:
        axs[1].set_xlabel('Time (s)')         
    
    axs[0].set_title('Before filter')
    axs[1].set_title('After filter')

    
    plt.savefig(fout, dpi = 300, fmt = 'png')
    plt.close('all')
    
    return



###################### MAIN ###################################################
if __name__ == '__main__':
    
    fld_in = 'E:/DATA/FourierToolbox/'
    fin = fld_in + 'testData.nii'
    freqbound = [0.01,0.1]
    TR = 2
      
    # IMPORT NIFTI
    img_in = nib.load(fin)
    DATA4D = img_in.get_fdata()
    Tpoints = np.arange(DATA4D.shape[-1]) * TR

    
    # GO TO FOURIER SPACE
    DATAFT = np.fft.rfft(DATA4D, axis = 3)
    DATAFTFREQ = np.fft.rfftfreq(DATA4D.shape[3], TR)

    masks = init_masks()

    # FILTER WITH THE CHOSEN WINDOW
    wm = 'Hanning'
    DATAFTFILTER = DATAFT * masks[wm]
    
    fout = fld_in + 'ftfilter_example.png'
    plot_filter(DATAFTFREQ,DATAFT,DATAFTFILTER, masks, wm, fout, in_fourier = True)
    
    # GO BACK TO REAL SPACE
    DATA4DFILTER = np.fft.irfft(DATAFTFILTER, axis=3)

    fout = fld_in + 'realfilter_example.png'
    plot_filter(Tpoints,DATA4D,DATA4DFILTER, masks, wm, fout, in_fourier = False)
    
    # SAVE AS NIFTI
    img_out = nib.Nifti1Image(DATA4DFILTER, np.eye(4))

    fout = 'E:/DATA/FourierToolbox/testDataFilter.nii'
    img_out.to_filename(fout)
    

    