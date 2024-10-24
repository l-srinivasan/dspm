#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri Sep 20 15:18:23 2024
@author: Leela Srinivasan

"""

# Import computational modules
import random
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests


def granger_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """
    

    Parameters
    ----------
    data : mne object
        time series data contained in stc.data
    variables : array
        array of strings for causation matrix labels.
    test : string, optional
        test type. The default is 'ssr_chi2test'.
    verbose : Boolean, optional
        The default is False.

    Returns
    -------
    df : df
        causation df/matrix
        
    """ 
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    MAXLAG=3
    
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=MAXLAG, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(MAXLAG)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
            
    #Label causation matrix
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df


def granger_causation_proportion(wm_df, stc, start_window, end_window):
    """
    

    Parameters
    ----------
    wm_df : df
        df of sensors involved in proposed white matter info.
    stc : mne object
        source timecourse.
    start_window : int
        start window for AUC integration.
    end_window : int
        end window for AUC integration.

    Returns
    -------
    granger_causal_proportion : int
        proportion of white matter connections that are Granger causal
        
    """  
    timeseries_df = pd.DataFrame()
    non_causal = 0
    for row_num in range(wm_df.shape[0]):
        timeseries_df['source_stc'] = stc.data[wm_df['Source Sensor'][row_num], start_window:end_window]
        timeseries_df['sink_stc'] = stc.data[wm_df['Destination Sensor'][row_num], start_window:end_window]
        
        mat = granger_causation_matrix(timeseries_df, variables = timeseries_df.columns)
        causality = mat.loc['sink_stc_y', 'source_stc_x']
        if causality > 0.00001:
            non_causal += 1

    granger_causal_proportion = 1-(non_causal/wm_df.shape[0])
    return granger_causal_proportion
    
    
def chance_causation_proportion(stc, start_window, end_window):
    """
    

    Parameters
    ----------
    stc : mne object
        source timecourse.
    start_window : int
        start window for AUC integration.
    end_window : int
        end window for AUC integration.

    Returns
    -------
    prop : int
        proportion of chance pairings that are Granger Causal
        
    """  
    #Initialize empty variables
    chance_prop=[]
    chance_timeseries_df=pd.DataFrame()
    
    #Set params
    source_virtual_sensors=5124
    significant_value_cutoff=0.00001
    chance_iterations=1000
    
    #Iterate through desired number to test chance
    for iter in np.arange(chance_iterations):  
        
        #Randomly select virtual sensor timecourse
        sensor1 =random.randint(0,source_virtual_sensors-1)
        sensor2 =random.randint(0,source_virtual_sensors-1)
        if sensor1 != sensor2:
            chance_timeseries_df['source_stc'] = stc.data[sensor1, start_window:end_window]
            chance_timeseries_df['sink_stc'] = stc.data[sensor2, start_window:end_window]
            
            #Compute chance and causality in both directions
            chance_mat = granger_causation_matrix(chance_timeseries_df, variables = chance_timeseries_df.columns)
            chance_xy_causality = chance_mat.loc['sink_stc_y', 'source_stc_x']
            chance_yx_causality = chance_mat.loc['source_stc_y', 'sink_stc_x']
            
            #Add result to list
            if chance_xy_causality > significant_value_cutoff or chance_yx_causality > significant_value_cutoff:
                chance_prop.append(0)
            else:
                chance_prop.append(1)
    
    #Calculate proportion of 1000 iterations
    prop=chance_prop.count(1)/(chance_prop.count(1)+chance_prop.count(0))
    return prop
