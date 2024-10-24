#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:40:10 2024
@author: srinivasanl2
"""


import os
import numpy as np
import pandas as pd
import mne
from mne.preprocessing import ICA


def load_run_and_find_trans(subject, run, session, freesurfer_dir, meg_subj_dir):
    """

    Parameters
    ----------
    subject : string
        Subject.
    run : int
        MEG Run.
    session : string
        clinical/altclinical/research.
    freesurfer_dir : string
        path to freesurfer folder created via the full processing stream for MR data
    meg_subj_dir : string
        path to raw CTF MEG files.

    Returns
    -------
    raw : mne object
        loaded CTF data.
    trans : fif file
        transformation from fiducial plane.

    """
    
    #Load run and make transformation matrix
    locate_run=[f for f in os.listdir(meg_subj_dir) if 'epilepsy' and '0{}-c.ds'.format(run) in f]    
    ds_path=os.path.join(meg_subj_dir, locate_run[0])
    raw = mne.io.read_raw_ctf(ds_path, system_clock = 'ignore', preload = False) # reading in CTF file that has been marked for defined subject
    run_trans_name=f'sub-{subject}_{session}-trans_0{run}.fif'
    
    #Locate trans depending on session
    if os.path.exists(os.path.join(freesurfer_dir,subject,'bem',run_trans_name)):
        trans = f'{freesurfer_dir}/{subject}/bem/{run_trans_name}'
    else:
        trans = f'{freesurfer_dir}/{subject}/bem/sub-{subject}_{session}-trans.fif'  
       
    #Select MEG channels
    raw.load_data() 
    raw.pick_types(meg = True, eeg = False, ref_meg = False) 
    
    return raw, trans


def downsample_and_filter(raw):
    """

    Parameters
    ----------
    raw : mne object
        loaded CTF MEG channels.

    Returns
    -------
    raw : mne object
        downsampled and filtered CTF data.
    Fs : int
        sampling rate.

    """
    
    if not raw.info['sfreq'] == 600:
        raw.resample(600) 
    Fs = raw.info['sfreq']
    raw.filter(5,50) 
    
    #Select bad channels
    raw.plot(clipping = None, n_channels = 50) 

    #Store bad channels
    if len(raw.info['bads']) >= 1:
        bad_meg = raw.info['bads'] 
        raw.pick_types(meg = True, exclude = bad_meg) 
        
    return raw, Fs


def ica(raw):
    """
    

    Parameters
    ----------
    raw : mne object
        downsampled and filtered CTF data.

    Returns
    -------
    raw : mne object
        downsampled and filtered CTF data.
    ica : mne object
        decomposed MEG signal into 30 independent components.

    """
    
    ica = ICA(method = 'fastica',
        random_state = 97,
        n_components = 30,
        verbose=True
        )
    ica.fit(raw,
        verbose = True,
        reject_by_annotation = True)
    ica.plot_sources(raw,title='ICA') 
    
    return raw, ica


def bem_src_forward(subject,freesurfer_dir, raw, trans):
    """
    

    Parameters
    ----------
    subject : string
        DESCRIPTION.
    freesurfer_dir : string
        path to freesurfer folder created via the full processing stream for MR data
    raw : mne object
        downsampled and filtered CTF data.
    trans : TYPE
        DESCRIPTION.

    Returns
    -------
    src : mne object
        description of the spatial geometry of the source space (ico.
    forward : TYPE
        DESCRIPTION.
    raw : mne object
        downsampled and filtered CTF data.

    """
    
    CONDUCTIVITY=0.3
    SURFACE='white'
    RES='ico4'
    
    
    #Create the BEM to model volume conduction and help with source space setup
    bem = mne.make_bem_model(
                             subject = subject, 
                             subjects_dir = freesurfer_dir, 
                             conductivity = [CONDUCTIVITY]
                             ) 
    bem_sol = mne.make_bem_solution(bem) #from model making solution that returns the model, will be used to inform source space setup


    # Set up source space with all candidate dipole locations using ICO4 framework
    src = mne.setup_source_space(
                                 subject = subject, 
                                 surface = SURFACE,
                                 spacing = RES, # USER INPUT - change based on desired resolution of source space
                                 subjects_dir = freesurfer_dir, 
                                 add_dist = True
                                 ) 

    #Compute the forward solution
    n_jobs = 4
    forward = mne.make_forward_solution(
                                        info = raw.info, 
                                        trans = trans, 
                                        src = src, 
                                        bem = bem_sol, 
                                        meg = True, 
                                        eeg = False, 
                                        )
    src = forward['src']
    return src, forward, raw



def create_epochs(raw, events, event_id):
    """
    

    Parameters
    ----------
    raw : mne object
        downsampled and filtered CTF data.
    events : TYPE
        CTF marks.
    event_id : TYPE
        DESCRIPTION.

    Returns
    -------
    epochs : mne object
        DESCRIPTION.
    baseline_epochs : mne object
        DESCRIPTION.
    noise_cov : TYPE
        DESCRIPTION.

    """
    
    MRK='S'
    BASELINE='B'
    TMIN=-1.5
    TMAX=0.5
    
    #Create spiking and baseline epochs and compute noise covariance
    epochs = mne.Epochs(raw, 
                        events, 
                        event_id = event_id[MRK], # USER INPUT # 
                        tmin = TMIN, 
                        tmax = TMAX, 
                        baseline = None,
                        proj = False,
                        reject_by_annotation = None,
                        preload = True
                        ) 
    
    
    baseline_epochs = mne.Epochs(raw, 
                            events, 
                            event_id = event_id[BASELINE], # USER INPUT # 
                            tmin = TMIN, 
                            tmax = TMAX, 
                            baseline = None
                            )
    
    
    noise_cov=mne.compute_covariance(baseline_epochs)
    
    return epochs, baseline_epochs, noise_cov
    

def estimate_evoked_responses(epochs):
    """
    

    Plot the joint response and scalp topographies.
    Parameters
    ----------
    epochs : mne object
        DESCRIPTION.

    Returns
    -------
    None.

    """
    epochs.average().plot_joint
    epochs.average().plot_topo(color="r", legend=False)
    
    
def apply_dspm(raw, forward, noise_cov, epochs):
    """
    

    Parameters
    ----------
    raw : mne object
        downsampled and filtered CTF data.
    forward : TYPE
        DESCRIPTION.
    noise_cov : TYPE
        DESCRIPTION.
    epochs : TYPE
        DESCRIPTION.

    Returns
    -------
    stc : mne object
        source reconstructed time series data.
    inv : mne object
        dSPM solution.

    """
    
    DEPTH=0.8
    METHOD = "dSPM"
    SNR = 3.0
    LAMBDA2 = 1.0 / SNR ** 2 

    inv = mne.minimum_norm.make_inverse_operator(info=raw.info, 
                                                    forward=forward, 
                                                    noise_cov=noise_cov, 
                                                    loose='auto', 
                                                    depth=DEPTH, 
                                                    fixed='auto',
                                                    rank=None, 
                                                    use_cps=True
                                                    )

    #Average epochs and create a moving average
    avg_epoch=epochs.average()
    stc=mne.minimum_norm.apply_inverse(avg_epoch,inv,LAMBDA2,METHOD,label=None)
    
    return stc, inv

def save_moving_average(stc, dspm_run_dir):
    """
    

    Parameters
    ----------
    stc : mne object
        source reconstructed time series data.
    dspm_run_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    stc : mne object
        moving average of source reconstructed time series data.

    """
    
    WINDOW=30
    SAMPLES=1201
    
    test_smoothed_stcs = np.zeros((1,SAMPLES))
    data_arr=stc.data
    row,column=data_arr.shape
    for tp in np.arange(0,row):
        tp_data=pd.DataFrame(stc.data[tp])
        moving_avg_source=[]
        moving_avg_source = tp_data.rolling(window = WINDOW, min_periods = 1, center = True).mean()
        moving_avg_source = moving_avg_source.to_numpy().T
        if tp == 0:
            test_smoothed_stcs=moving_avg_source
        else:
            test_smoothed_stcs = np.vstack([test_smoothed_stcs, moving_avg_source])

    #Save data
    stc.data=test_smoothed_stcs
    data_arr=stc.data
    np.save(os.path.join(dspm_run_dir,'stc_array'),data_arr)
    os.chdir(dspm_run_dir)
    stc.save('full-stcs', overwrite=True)
    
    return stc


def save_parcel_vertex_mapping(subject, freesurfer_dir, stc):
    """
    

    Parameters
    ----------
    subject : string
        subject code
    freesurfer_dir : string
        path to freesurfer folder created via the full processing stream for MR data
    dspm_run_dir : TYPE
        DESCRIPTION.
    
    """
    
    #Read cortical parcellation labels from a FreeSurfer annotation file
    parcellation_type = 'Schaefer2018_200Parcels_7Networks_order'
    read_labels = mne.read_labels_from_annot('fsaverage', parc = parcellation_type, 
                                          subjects_dir=freesurfer_dir)
    subject_labels = mne.morph_labels(read_labels, subject_to = subject, subjects_dir = freesurfer_dir)


    # Iterate over each parcel in subject_labels, finding vertices corresponding to the label vertices
    parcel_vertex_mapping = {}
    for parcel_num, label in enumerate(subject_labels):
        idx = np.nonzero(np.isin(stc.vertices[0], label.vertices))[0]
        parcel_vertex_mapping[parcel_num] = idx.tolist()
    np.save("parcel_vertex_mapping.npy", parcel_vertex_mapping)
    
    
def save_vs_distances(src):
    """
    

    Parameters
    ----------
    src : MNE object
        source space

    Returns
    -------
    geodesic_distances : array
        list of two matrices containing the distances between virtual sensors
    """

    # Save distances between dense vertices in a sparse matrix
    print('Calculating distances between dense and subsampled vertices...')
    
    geodesic_distances=[]
    
    # Save distances between subsampled vertices/VS
    for hemi_index in [0, 1]:
        distances = src[hemi_index]['dist']
        vertno = src[hemi_index]['vertno']
        
        # Calculate distances between each virtual sensor pairing, populate matrix
        virtual_sensor_distances = np.empty([len(vertno), len(vertno)])
        for sensor_rownum in range(len(vertno)):
            for sensor_colnum in range(len(vertno)):
                virtual_sensor_distances[sensor_rownum, sensor_colnum] = distances[vertno[sensor_rownum], vertno[sensor_colnum]]
        
        # Save to CSV in dSPM run directory
        f = 'lh_VS_distances.csv' if hemi_index==0 else 'rh_VS_distances.csv'
        np.savetxt(f, virtual_sensor_distances, delimiter=",")
        
        geodesic_distances.append(virtual_sensor_distances)
    return geodesic_distances
        

def save_src(src, inv):
    """
    

    Parameters
    ----------
    src : mne object
        source space
    inv : mne object
        inverse source space
    
    """
    mne.write_source_spaces(fname = 'mymodel.fif', src = src, overwrite = True)
    mne.write_source_spaces(fname = 'inv_mymodel.fif', src = inv['src'], overwrite = True)
    

def save_dspm_movies(SUBJECT, src, inv, stc, mrk, freesurfer_dir):
    """
    

    Parameters
    ----------
    SUBJECT : TYPE
        DESCRIPTION.
    src : TYPE
        DESCRIPTION.
    inv : TYPE
        DESCRIPTION.
    stc : TYPE
        DESCRIPTION.
    mrk : TYPE
        DESCRIPTION.
    freesurfer_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """    
    INTERPOLATION='linear'
    TIME_DIL=50
    TMIN=-0.5
    TMAX=0.3
    FRAME_RATE=10
    
    src = inv['src']
    
    brain_lh_lat_evoked = stc.plot(src = src, 
                                    subject = SUBJECT, 
                                    subjects_dir = freesurfer_dir, 
                                    hemi = 'lh', 
                                    views = 'lateral',
                                    )

    brain_lh_lat_evoked.save_movie(filename=f'{mrk}_smoothed_lh_lat.mov',
                            time_dilation = TIME_DIL, 
                            tmin = TMIN, 
                            tmax = TMAX,
                            interpolation = INTERPOLATION, 
                            framerate = FRAME_RATE
                            )
    
    brain_lh_mes_evoked = stc.plot(src = src, 
                                        subject = SUBJECT, 
                                        subjects_dir = freesurfer_dir, 
                                        hemi = 'lh', 
                                        views = 'medial'
                                       ) 
    
    brain_lh_mes_evoked.save_movie(filename = f'{mrk}_smoothed_lh_mes.mov',
                            time_dilation = TIME_DIL, 
                            tmin = TMIN, 
                            tmax = TMAX,
                            interpolation = INTERPOLATION, 
                            framerate = FRAME_RATE
                            )
    
    brain_rh_lat_evoked = stc.plot(src = src, 
                                        subject = SUBJECT, 
                                        subjects_dir = freesurfer_dir, 
                                        hemi = 'rh', 
                                        views = 'lateral',
                                        ) 
    
    brain_rh_lat_evoked.save_movie(filename=f'{mrk}_smoothed_rh_lat.mov',
                            time_dilation = TIME_DIL, 
                            tmin = TMIN, 
                            tmax = TMAX,
                            interpolation = INTERPOLATION, 
                            framerate = FRAME_RATE
                            )
    
    brain_rh_mes_evoked = stc.plot(src = src, 
                                        subject = SUBJECT, 
                                        subjects_dir = freesurfer_dir, 
                                        hemi = 'rh', 
                                        views = 'medial',
                                        ) 
    
    brain_rh_mes_evoked.save_movie(filename = f'{mrk}_smoothed_rh_mes.mov',
                            time_dilation = TIME_DIL, 
                            tmin = TMIN, 
                            tmax = TMAX,
                            interpolation = INTERPOLATION, 
                            framerate = FRAME_RATE
                            )
    

def check_trans(subject, run, dspm_run_dir, subj_fs_dir):
    """
    

    Parameters
    ----------
    subject : TYPE
        DESCRIPTION.
    run : TYPE
        DESCRIPTION.
    dspm_run_dir : TYPE
        DESCRIPTION.
    subj_fs_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if os.path.isfile(os.path.join(dspm_run_dir, 'trans_mat.1D')):
        print("Transformation 1D file found in dSPM run directory. Proceeding with clustering...")
       
    else:
        written=False
        if 'bem' in os.listdir(subj_fs_dir):
            fs_bem = os.path.join(subj_fs_dir,'bem')
            
            for file in os.listdir(fs_bem):
                if 'trans_0{}'.format(run) in file:
                    trans_arr = mne.read_trans(os.path.join(fs_bem,file))['trans']
                    
                    if np.array_equal(trans_arr[3,:], np.array([0,0,0,1])):
                        trans_arr = np.delete(trans_arr, 3, 0)
                        np.savetxt(os.path.join(dspm_run_dir, 'trans_mat.1D'), trans_arr)
                        written=True
                        
                    else:
                        raise Exception('Manually verify fourth row of transformation matrix. Exiting...')
            
            if written==False:
                raise Exception('Transformation matrix not found/written. Exiting...')
                    
        
        else:
            raise Exception('Freesurfer folder for {} missing bem folder. Exiting...'.format(subject))
    
    
