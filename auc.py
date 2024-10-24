#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Sep 24 09:56:37 2024
@author: Leela Srinivasan

"""

import pandas as pd
import numpy as np
import scipy.stats as stats


def create_auc_window(stc):
    """
    

    Parameters
    ----------
    stc : mne object
        source reconstructed time series data.

    Returns
    -------
    Window bounds.

    """
    LOWER_LIM=2
    WIDTH=25
    
    #Find the timepoints of maximum values over full stc for each sensor
    vs_max_values = np.amax(stc.data, axis=1)
    time_at_max_values = np.argmax(stc.data, axis=1)
    
    # Take the median timepoint from the maximum values that exceed a zscore of 2
    zscored_max_times = time_at_max_values[stats.zscore(vs_max_values) > LOWER_LIM]
    median_timepoint = np.round(np.median(zscored_max_times))
        
    # Create a time window around the median timepoint using +/-25 timepoints
    start_window = int(median_timepoint)-WIDTH
    end_window = int(median_timepoint)+WIDTH
    return start_window, end_window


def threshold_stc(stc):
    """
    

    Parameters
    ----------
    stc : mne object
        source reconstructed time series data.

    Returns
    -------
    Thresholded stc.

    """
    BASELINE_WIDTH=500
    LOWER_LIM=2
    EXCEPTION_LIM=1
    
    # Create subset of timecourses for thresholding reference
    timecourses = pd.DataFrame(stc.data.copy())
    first_500 = pd.DataFrame(stc.data[:, 0:BASELINE_WIDTH])
    
    # Loop through sensor timecourses, calculate zscore on first 500 samples of rest/nonspiking data
    # Set threshold above zscore=2
    for vs in first_500.index:
        try:
            zscores = stats.zscore(first_500.loc[vs, :])
            first_occ_above_2 = zscores[zscores > LOWER_LIM].idxmin()
            threshold = first_500.loc[vs, first_occ_above_2]
        except:
            # There is no value in this timecourse for which the zscore is greater than 2 
            # Manually set a conservative threshold of amplitude 1 to handle exception
            threshold = EXCEPTION_LIM
        
        # Replace this row of the timecourses with the thresholded values
        # Values out of bounds will be reported as 0
        temp = pd.DataFrame(timecourses.loc[vs, :])
        thresholded_sensor = temp.where(temp>threshold, 0)
        timecourses.loc[vs] = thresholded_sensor[vs]
        
    # Save thresholded timecourses
    thresh_stc = timecourses.to_numpy()
    np.save('stc_thresholded_array.npy', thresh_stc)
    return thresh_stc


def integrate_stc(auc_stc, thresh_stc, start_window, end_window):
    """
    

    Parameters
    ----------
    auc_stc : mne object
        deep copy of stc object containing source reconstructed time series data.
    thresh_stc : mne object
        thresholded source reconstructed time series data.
    start_window : int
        lower bound for AUC integration window.
    end_window : int
        upper bound for AUC integration window.

    Returns
    -------
    auc : array
        AUC value for each of 5124 sensors
    percentile_cutoff : int
        auc value at the CUTOFF percentile (set to 95).
    """
    
    #Overwrite a deep copy of stc.data with area under the curve calculation for time series
    CUTOFF=95
    auc = np.trapz(thresh_stc[:,start_window:end_window],  axis = 1)
    auc_stc.crop(0,0)
    auc_stc.data[:,0] = auc
    
    # Save out AUC to be projected onto the cortical surface
    np.savetxt('auc_thresh_array', auc, delimiter=', ')
    auc_stc.save('auc_thresh', overwrite=True)
    
    # Save 95th percentile of AUC range to text file for clustering threshold
    percentile_cutoff=np.nanpercentile(auc,[CUTOFF])
    np.savetxt('percentile_cutoff', percentile_cutoff)
    np.save('window_timepoints.npy', [start_window, end_window])
    
    return auc, percentile_cutoff
