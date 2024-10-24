#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Fri Sep 20 15:16:24 2024
@author: Leela Srinivasan

"""

import os
import math
import shutil
import pandas as pd
import warnings


def suppress_warnings():
    """
    

    Returns
    -------
    None.

    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None
    def fxn():
        warnings.warn("deprecated", DeprecationWarning)
    with warnings.catch_warnings(action="ignore"):
        fxn()
        

def euclidean_distance(point1, point2):
    """
    

    Parameters
    ----------
    point1 : array
        XYZ coordinate.
    point2 : array
        XYZ coordinate.

    Returns
    -------
    Euclidean distance between points.

    """
    distance = 0.0
    for i in range(len(point1)):
        distance += (point2[i] - point1[i]) ** 2
    return math.sqrt(distance)# Define function to search for parcel containing sensor


def associated_key(val, dictionary):
    """
    

    Parameters
    ----------
    val : int/string
        value to search for within keys.
    dictionary : dict
        dictionary to search through.

    Returns
    -------
    Associated key.

    """
    for key, values in dictionary.items():
        if val in values:
            return key


def overlapping_windows(window1, window2):
    """
    

    Parameters
    ----------
    window1 : array
        window limits.
    window2 : array
        window limits.

    Returns
    -------
    Boolean describing whether windows overlap.

    """
    if (window1[0] < window2[1] and window1[1] > window2[0]) or (window2[0] < window1[1] and window2[1] > window1[0]): 
        return True
    return False


def clean_wdir():
    """
    
    Remove unncessary files

    Returns
    -------
    None.

    """
    for file in os.listdir():
        if 'std.60' in file:
            os.remove(file)
        elif 'myhead' in file:
            os.remove(file)
            
            
def organize_wdir():
    """
    
    Sort files

    Returns
    -------
    None.

    """
    
    os.mkdir("../projected_clusters")
    for file in os.listdir():
        if '.nii' in file:
            shutil.move(file, "../projected_clusters/"+file)
            
    os.mkdir("../source_reconstruction")
    for file in os.listdir():
        for substring in ['full-stcs', 'VS_distances', 'mymodel', 'parcel_vertex_mapping', 'suma2mne']:
            if substring in file:
                shutil.move(file, "../ied_source_reconstruction/"+file)
                

def locate_freesurfer_folder(subj, freesurfer_dir):
    """

    Parameters
    ----------
    subj : string
        p***.
    freesurfer_dir : string
        path to freesurfer directory for all subjects.

    Returns
    -------
    subj_freesurfer_dir : string
        path to freesurfer directory for specific subject.
    SUMA_dir : string
        path to freesurfer SUMA directory for specific subject.
    session : string
        clinical/altclinical.

    """
    for file in os.listdir(freesurfer_dir):
        if subj+'_' in file:
            subj_freesurfer_dir=os.path.join(freesurfer_dir,file)
            SUMA_dir = os.path.join(subj_freesurfer_dir, 'SUMA')
            if 'alt' in file:
                session='ses-altclinical'
            else:
                session='ses-clinical'
        if subj==file:
            print('Reset freesurfer naming convention.')
    return subj_freesurfer_dir, SUMA_dir, session


def switch_to_bids(subj, session, freesurfer_dir):
    """
    

    Parameters
    ----------
    subj : string
        p***.
    session : string
        clinical/altclinical.
    freesurfer_dir : string
        path to freesurfer directory for all subjects.

    Returns
    -------
    None.

    """
    if not os.path.exists(os.path.join(freesurfer_dir,subj)):
        os.rename(os.path.join(freesurfer_dir,'sub-'+subj+'_'+session),os.path.join(freesurfer_dir,subj))
