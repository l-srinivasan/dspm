#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon Sep 23 15:07:55 2024
@author: Leela Srinivasan

Usage: ./main.py p*** CTF_RUN [PLOT_FUNCS=True]
Example: ./main.py p000 3 False (plotting suppressed)

Requirements: 
    Modules: mne-python, stc2gii_hack, nih2mne
    Programs: AFNI/SUMA
    Pre-processing: Freesurfer recon-all, Boundary Element Model/Tranformation Matrix creation
    
"""

import sys
import os
import copy
import subprocess


import pandas as pd
import numpy as np
import mne
import mne.channels
mne.set_log_level(False)


# Read in subject data through command line args
SUBJECT=sys.argv[1]
RUN=sys.argv[2]
try:
  PLOT_FUNCS = sys.argv[3]
except IndexError:
  PLOT_FUNCS = True
CLINICIAN_MARKER = 'S'


# Import helper functions
import config
import general, plotting, dspm, auc
import cluster, wm, causation, resection
general.suppress_warnings()


#Set user + data directories
neu_dir='/Volumes/shares/NEU'
freesurfer_dir=os.path.join(neu_dir,'Data/derivatives/freesurfer-6.0.0')
ctf_dir=os.path.join(neu_dir,'Projects/CTF_MEG')
dspm_dir=os.path.join(ctf_dir,SUBJECT,'dSPM')


#Create working directory
run_dir=os.path.join(dspm_dir,'run_0{}'.format(RUN))
if not os.path.exists(run_dir):
    os.mkdir(run_dir)
wdir=os.path.join(run_dir,'surface_clusters')
if not os.path.exists(wdir):
    os.mkdir(wdir)
os.chdir(wdir)
    
    
# Set personal directories
user_dir="/Volumes/Shares/NEU/Users/Leela"
code_dir=os.path.join(user_dir, 'paper', 'code')
user_data_dir = os.path.join(user_dir,'data')


#Freesurfer setup: Move from bids naming convention to MNE compatible folder
subj_freesurfer_dir, SUMA_dir, SESSION = general.locate_freesurfer_folder(SUBJECT, freesurfer_dir)
freesurfer_prefix='sub-'+SUBJECT+'_'+SESSION
general.switch_to_bids(SUBJECT, SESSION, freesurfer_dir)


#Internal MEG code conversion
f="/Volumes/shares/NEU/Scripts_and_Parameters/meg_key"
df = pd.read_csv(f, delimiter = "=", header = None, names = ['meg','pnum'])
subj_key_dict = dict(zip(df.pnum,df.meg))
meg_subj_name = subj_key_dict[SUBJECT]


#Locate desired raw MEG run data, load, downsample, and filter. Select channels for exclusion, run ICA
meg_subj_dir = os.path.join(ctf_dir,SUBJECT,'CTF')
raw, trans = dspm.load_run_and_find_trans(SUBJECT, RUN, SESSION, freesurfer_dir, meg_subj_dir) 
raw, Fs = dspm.downsample_and_filter(raw)
events, event_id = mne.events_from_annotations(raw) 
raw, ica = dspm.ica(raw)


#BEM, src, forward solution, epoching
src, forward, raw = dspm.bem_src_forward(SUBJECT,freesurfer_dir, raw, trans)
geodesic_distances = dspm.save_vs_distances(src)


#Create epochs, apply dSPM
epochs, baseline_epochs, noise_cov = dspm.create_epochs(raw, events, event_id)
dspm.estimate_evoked_responses(epochs)
stc, inv= dspm.apply_dspm(raw, forward, noise_cov, epochs)
stc=dspm.save_moving_average(stc, wdir)
dspm.save_parcel_vertex_mapping(SUBJECT, freesurfer_dir, stc)


#Save out src for SUMA conversion, restore BIDS naming convention
dspm.save_src(src, inv)
#dspm.save_dspm_movies(SUBJECT, src, inv, stc, 'S', freesurfer_dir)
os.rename(os.path.join(freesurfer_dir,SUBJECT),os.path.join(freesurfer_dir,'sub-'+SUBJECT+'_'+SESSION))


#Calculate area under the curve
start_window, end_window = auc.create_auc_window(stc)
if PLOT_FUNCS: plotting.plot_butterfly_timeseries(stc, start_window, end_window)
thresh_stc=auc.threshold_stc(stc)
auc_stc=copy.deepcopy(stc)
auc, percentile_cutoff=auc.integrate_stc(auc_stc, thresh_stc, start_window, end_window)


#Copy SUMA files to auc directory
SUMA_dir=os.path.join(subj_freesurfer_dir, 'SUMA')
cluster.copy_surface_volume(SUBJECT, SESSION, SUMA_dir, wdir)
cluster.copy_standard_meshes(SUMA_dir, wdir)


#Check for fiducial transformation matrix and cluster AUC
dspm.check_trans(SUBJECT, RUN, wdir, subj_freesurfer_dir)
cmd='sh {} {}'.format(os.path.join(code_dir, 'cluster_auc.sh'), freesurfer_prefix)
subprocess.run(cmd,shell=True)
PRIMARY_HEMI=cluster.determine_primary_cluster()


# Create Schaefer network map
file='schaefer_400_parcel_labels.pickle'
f = os.path.join(user_data_dir, file)
network_map, parcel_names = wm.map_parcels_to_networks(f)


# Compile information across clusters for analysis
COUNTER=1
suma2mne_list=[]
for HEMI in "lh", "rh":
    
    #Generate mapping of SUMA nodes to MNE source space
    suma2mne=cluster.map_SUMA_to_mne(SUBJECT, HEMI)
    suma2mne_list.append(suma2mne)
    suma2mne= suma2mne.set_index('SUMA_vertex')
    SUMA_cluster_map,clusters=cluster.find_clst_mask(HEMI)
    
    
    # Add in AUC information
    for ind, auc_val in enumerate(auc):
        suma2mne.loc[suma2mne['virtual_sensor_node'] == float(ind), 'auc'] = auc_val

    
    # Project surface clusters into the volume; merge surface files for combined viz
    cluster.project_clusters(SUBJECT, SESSION, HEMI, wdir, subj_freesurfer_dir, clusters, SUMA_cluster_map, suma2mne)
    
    
    # Create master df with information on all clusters
    MASTER_CREATED=False
    while True:
        f='{}_{}_virtual_sensor_auc.csv'.format(COUNTER, HEMI)
        if os.path.exists(os.path.join(wdir, f)):
            temp_df=cluster.compile_cluster_data(COUNTER, HEMI, stc, start_window, end_window, geodesic_distances, PLOT_FUNCS)

            if MASTER_CREATED==False:
                master_df=temp_df.copy()
                MASTER_CREATED=True
            master_df = master_df.merge(temp_df, how='outer')
            COUNTER+=1
            
        else:
            COUNTER=1 #Reset for other hemisphere
            break
        
    if HEMI==PRIMARY_HEMI:
        primary_df=master_df.copy()


# Load parcel mapping dictionary / create if it doesn't exist
f = '{}_parcel_vertex_mapping.npy'.format(PRIMARY_HEMI)
if not os.path.exists(os.path.join(wdir, f)):
    wm.generate_parcel_vertex_mapping(freesurfer_dir, wdir, stc, SUBJECT, SESSION)


# Find associated parcel/Yeo network for each virtual sensor. Exclude the medial Wall
parcel_vertex_map = np.load(os.path.join(wdir, f), allow_pickle=True).item()
primary_df['parcel'] = [general.associated_key(vs, parcel_vertex_map) for vs in primary_df.virtual_sensor.values]
primary_df['network'] = [general.associated_key(parcel_names[parcel], network_map) for parcel in primary_df.parcel.values]
primary_df = primary_df[primary_df.network != 'Medial Wall']
if PLOT_FUNCS: plotting.plot_cluster_time_across_threshold(primary_df)


# Reorder clusters based on timing; remove activity after last cluster peaks, merge SUMA location
primary_df=cluster.reorder_clusters(primary_df, stc)
sliced_df=cluster.slice_df(primary_df)
suma2mne= suma2mne_list[0] if PRIMARY_HEMI=='lh' else suma2mne_list[1]
sliced_df=cluster.merge_suma2mne(sliced_df, suma2mne)


#Find white matter connections
wm_info = wm.find_connections(sliced_df, parcel_names)
columns=['Source Parcel', 'Source Sensor', 'Destination Parcel', 'Destination Sensor', 'Cluster Pair', 'Parcel Names', 'Network', 'Within Network']
wm_df = pd.DataFrame(wm_info, columns=columns)
wm.mark_source_and_sink_activity(wm_df, 1, 2, suma2mne, PRIMARY_HEMI)


#Run Granger causal and chance analysis by iterating through white matter connections
granger_causal_proportion = causation.granger_causation_proportion(wm_df, stc, start_window, end_window)
chance_proportion=causation.chance_causation_proportion(stc, start_window, end_window)


#Get resection/cluster overlap statistics
f='rsxn.msk.nii'
resection.import_resection(config)
resection.align_resection_to_surfvol(SUBJECT, SESSION, f)
rsxn_cm=resection.get_resection_cm(f)
output_list=resection.get_resection_cluster_overlap(SUBJECT, rsxn_cm)
resection.print_overlap_results(output_list)


#Clean up working directory
general.clean_wdir()
general.organize_wdir()
