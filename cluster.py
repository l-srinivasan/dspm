#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon Sep 23 13:40:38 2024
@author: Leela Srinivasan

"""

import os
import glob
import shutil
import scipy.stats
import pandas as pd
import numpy as np
from sh import gunzip
import matplotlib.pyplot as plt
import plotting
import subprocess


def copy_surface_volume(subject, session, SUMA_dir, wdir):
    """
    

    Parameters
    ----------
    subject : string
        p***.
    session : string
        clinical/altclinical.
    SUMA_dir : string
        path to freesurfer SUMA directory for specific subject.
    wdir : string
        path to working directory/surface_clusters directory.

    Returns
    -------
    None.

    """
    surf_vol='sub-{}_{}_SurfVol.nii'
    surf_vol=surf_vol.format(subject, session)
    shutil.copyfile(os.path.join(SUMA_dir, surf_vol), os.path.join(wdir, surf_vol))
    

def copy_standard_meshes(SUMA_dir, wdir, MESH=60):
    """
    

    Parameters
    ----------
    SUMA_dir : string
        path to freesurfer SUMA directory for specific subject.
    wdir : string
        path to working directory/surface_clusters directory.
    MESH : int (optional)
        Freesurfer standard cortical mesh type. The default is 60.

    Returns
    -------
    None.

    """
    standard_meshes=[f for f in os.listdir(SUMA_dir) if f.startswith('std.{}'.format(MESH))]
    for std in standard_meshes:
        shutil.copyfile(os.path.join(SUMA_dir, std), os.path.join(wdir, std))
        
        
def map_SUMA_to_mne(subject, hemi):
    """
    

    Parameters
    ----------
    subject : string
        p***.
    hemi : string
        hemisphere (lh/rh).

    Returns
    -------
    suma2mne : df
        map between SUMA XYZ vertices on the cortical surface to the MNE source space

    """
    #Read in the XYZ coordinates of each SUMA node
    f='myhead-'+hemi+'_inv_shifted.gii.coord.1D.dset'
    node_locations=pd.read_csv(f,skiprows=9,header=None,sep='\t')
    cols={0:'virtual_sensor_node',1:'SUMA_X',2:'SUMA_Y',3:'SUMA_Z',4:'tmp'}
    node_locations=node_locations.rename(columns=cols)
    node_locations=node_locations.drop(columns=['tmp'])

    #Read in corresponding virtual sensors for each vertex
    vertex_to_node=pd.read_csv('std.60.'+hemi+'.1D',skiprows=16,delim_whitespace=True)
    cols={'#0':'vertex','#1':'First node','#2':'Second node','#3':'Third node','#4':'First node weight','#5':'Second node weight','#6':'Third node weight'}
    vertex_to_node=vertex_to_node.rename(columns=cols)
    
    #Initialize conversion df
    out_df=pd.DataFrame(columns=['SUMA_vertex','virtual_sensor_node','SUMA_X','SUMA_Y','SUMA_Z'])
    for vertex in vertex_to_node['vertex']:
        node=float(vertex_to_node['First node'][vertex_to_node[vertex_to_node['vertex']==vertex].index.values])
        if not node == -1:
            SUMA_X=node_locations['SUMA_X'][node]
            SUMA_Y=node_locations['SUMA_Y'][node]
            SUMA_Z=node_locations['SUMA_Z'][node]
            tmp_dict={'SUMA_vertex':vertex,'virtual_sensor_node':node,'SUMA_X':SUMA_X,'SUMA_Y':SUMA_Y,'SUMA_Z':SUMA_Z}
            tmp_df=pd.DataFrame(tmp_dict,index=[node])
            out_df=pd.concat([out_df,tmp_df])

    #Drop index and write to CSV
    out_df=out_df.reset_index()
    out_df=out_df.drop(columns=['index'])
    outname=hemi+'_suma2mne.csv'
    out_df.to_csv(outname,sep=' ',index=False)
    return out_df

        
def determine_primary_cluster():
    """
    

    Returns
    -------
    hemi : string
        hemisphere containing primary cluster.

    """
    area=0
    f_suffix='_CLUSTERED_ClstTable_e2_n100.1D'
    for hemi in 'lh', 'rh':
        
        clst_table=np.genfromtxt(hemi+f_suffix)
        if len(clst_table.shape) > 1: #If there is more than one cluster on this hemi
            temp_area=clst_table[0][2]
        else:
            temp_area=clst_table[2]
        
        if hemi=='rh':
            if temp_area>area:
                return hemi
            else:
                return 'lh'
            
        area=temp_area
    return None
     
    
def find_clst_mask(hemi):
    """
    

    Parameters
    ----------
    hemi : string
        hemisphere containing primary cluster.

    Returns
    -------
    SUMA_cluster_map : df
        map of which cluster each SUMA node belongs to
    clusters : array
        array of clusters on the corresponding hemisphere
        
    """    
    for file in os.listdir():
        if hemi in file and 'ClstMsk' in file:
            SUMA_cluster_map=(pd.read_csv(file,skiprows=12,header=None,sep=' ',names=['tmp','node','val'],nrows=36002)).drop(columns=['tmp'])
            uniq = SUMA_cluster_map.val.unique()
            clusters = [val for val in uniq if val!=0]
            return SUMA_cluster_map, clusters
    return None, None


    
# Define function to compile cluster attributes
def compile_cluster_data(cluster_number, hemi, stc, start_window, end_window, geodesic_distances, PLOT_FUNCS):
    """
    

    Parameters
    ----------
    cluster_number : int
        cluster number.
    hemi : string
        hemisphere containing primary cluster.
    stc : mne object
        source reconstructed time series data.
    start_window : int
        lower bound for AUC integration window.
    end_window : int
        upper bound for AUC integration window.
    geodesic_distances : 2d array
        distances between each pairing of MNE source reconstructed virtual sensors.
    PLOT_FUNCS : Boolean
        Indication to plot.

    Returns
    -------
    df : df
        cleaned df for cluster with timing across threshold added

    """    
    OUTLIER_BOUND=3
    SENSORS_PER_HEMI=2562
    ORDER=1
    THRESHOLD_PERCENTAGE=0.75
    DISCRETE_TO_MS=5/3
    
    """
    AUC
    """
    
    # Load file containing sensors in the primary cluster and remove outliers
    f = '{}_{}_virtual_sensor_auc.csv'.format(cluster_number, hemi)
    df = pd.read_csv(f, delim_whitespace = True)
    df = df[(np.abs(scipy.stats.zscore(df.auc)) < OUTLIER_BOUND)]
    
    # Find vs with max AUC and assign color for graphing
    max_auc_vs = int(df.sort_values('auc').iloc[-1]['virtual_sensor'])
    df['color'] = 'grey'
    df.loc[df.virtual_sensor == max_auc_vs, 'color'] = 'red'

    """
    GREY MATTER
    """
    
    # Find distance from max AUC & plot max AUC
    df['distance'] = [geodesic_distances[0][max_auc_vs, vs]*1000 if hemi=='lh' else geodesic_distances[1][max_auc_vs-SENSORS_PER_HEMI, vs-SENSORS_PER_HEMI]*1000 for vs in df.virtual_sensor.values]
    P_VALUE=scipy.stats.pearsonr(df.distance, df.auc)[1]
    ROUND_P_VALUE=f"{P_VALUE:.2e}"
    
    if PLOT_FUNCS:
        scatter_points = plt.scatter(df.distance, df.auc, c='deepskyblue')
        max_auc_row = df[df.color == 'red']
        maximal_auc = plt.scatter(max_auc_row.distance.values[0], max_auc_row.auc.values[0], c='blueviolet')
        
        # Plot trendline
        z = np.polyfit(df.distance, df.auc, ORDER)
        p = np.poly1d(z)
        trend_line = plt.plot(df.distance, p(df.distance), c='deepskyblue')
    
        plt.title('Area Under the Curve vs. Distance, {} Cluster {}'.format(hemi, cluster_number))
        plt.xlabel('Distance from Virtual Sensor with Maximal AUC (mm)')
        plt.ylabel('AUC (dSPM units)')
        plt.legend([scatter_points, maximal_auc, trend_line[0]], ['Virtual Sensors', 'Maximal AUC VS', 'p={}'.format(ROUND_P_VALUE)])
        plt.show()

    """
    WHITE MATTER
    """
    
    # Assign threshold to be 75% of the dSPM amplitude of the timecourse with maximal AUC
    threshold = stc.data[max_auc_vs,start_window:end_window].max()*THRESHOLD_PERCENTAGE
    
    # Shift window to include timepoints from when the sensor with the maximal AUC crosses the threshold
    max_crosses_threshold_tp = np.argmax(stc.data[max_auc_vs, :] > threshold)
    if max_crosses_threshold_tp < start_window:
        start_window = max_crosses_threshold_tp
    
    # Find index where stc crosses threshold within AUC window
    above_threshold = stc.data[df.virtual_sensor.values, start_window:end_window] > threshold
    crossing_point = [np.argmax(row) if True in row else np.NaN for row in above_threshold]
    df['time_across_threshold'] = [(x + start_window)* DISCRETE_TO_MS for x in crossing_point]
    
    #Remove sensors that don't cross threshold within the window and reorder by time
    df.time_across_threshold = df.time_across_threshold.replace(0, np.nan)
    df = df[df['time_across_threshold'].notna()]
    df = df.sort_values('time_across_threshold').reset_index(drop=True)
    df['cluster'] = int(cluster_number)
    if PLOT_FUNCS: plotting.plot_time_across_threshold(stc, df, threshold)
    
    return df


def reorder_clusters(df, stc):
    """
    

    Parameters
    ----------
    df : df
        compiled cluster statistics
    stc : mne object
        source reconstructed time series data.

    Returns
    -------
    df : df
        df reordered based on timing

    """
    num_clusters=len(df['cluster'].unique())
    if num_clusters < 2:
        raise Exception("Subject has one cluster. Skipping white matter analysis.")
    df = df.sort_values('time_across_threshold').reset_index(drop=True)
    reorder_clusters = dict(zip(df.loc[df.color == 'red'].cluster.values, 
                                list(range(1, num_clusters+1))))
    df = df.replace({"cluster": reorder_clusters})
    plotting.plot_cluster_timeseries(stc, df)
    return df


def slice_df(df):
    """
    

    Parameters
    ----------
    df : df
        df reordered based on timing.

    Returns
    -------
    sliced_df : df
        df sliced based on sensors that could be involved in white matter connections.

    """
    slicing_idx = df.loc[df.color == 'red'].iloc[-1].name + 1
    sliced_df = df[:slicing_idx]
    return sliced_df


def merge_suma2mne(df, suma2mne):
    """
        

    Parameters
    ----------
    df : df
        df with virtual sensor info
    suma2mne : df
        df with relationship between SUMA XYZ nodes and MNE source reconstructed space.

    Returns
    -------
    df : df
        df with SUMA info and virtual sensor info

    """
    suma2mne['virtual_sensor'] = suma2mne.virtual_sensor_node.astype(int)
    df = df.merge(suma2mne, how='inner', on='virtual_sensor').drop_duplicates(subset=['virtual_sensor']).reset_index(drop=True)
    return df


def project_clusters(SUBJECT, session, hemi, wdir, subj_freesurfer_dir, clusters, SUMA_cluster_map, suma2mne, NROWS=36002):
    """
    
    Project the clusters into the volume for resection mask volumentric comparison and AFNI viewing.
    Create files with merged clusters.
    
    
    Parameters
    ----------
    SUBJECT : string
        p***.
    session : string
        clinical/altclinical.
    hemi : string
        hemisphere containing primary cluster.
    wdir : string
        path to working directory/surface_clusters directory.
    subj_freesurfer_dir : string
        path to freesurfer directory for specific subject.
    clusters : array
        array of clusters on the corresponding hemisphere
    SUMA_cluster_map : df
        mapping describing which cluster each SUMA node belongs to.
    suma2mne : df
        df with relationship between SUMA XYZ nodes and MNE source reconstructed space.
    NROWS : int, optional
        Number of SUMA nodes. The default is 36002.

    Returns
    -------
    None.

    """    
    # Populate missing SUMA nodes with zeroes
    suma_df=pd.DataFrame(np.arange(0,NROWS),columns=['auc'])
    suma_df.loc[suma2mne.index, 'auc'] = suma2mne.auc
    
    # Mask nodes with color coded clusters for viz purposes instead of AUC values
    colored_clust_df = pd.DataFrame(0, index = np.arange(NROWS), columns = ['cluster'])
    
    if not clusters:
        return None
    
    for cluster in clusters:
        
        # Mask nodes to those contained in the cluster
        cluster_suma_df = suma_df.copy()
        cluster_suma_df.loc[SUMA_cluster_map[SUMA_cluster_map.val != cluster].node, :] = 0
        
        # Mask nodes with color coded clusters instead of AUC vals
        tmp_colored_clust = cluster_suma_df.copy()
        tmp_colored_clust.loc[tmp_colored_clust.auc > 0] = -1*cluster
        colored_clust_df['cluster'] = colored_clust_df['cluster'] + tmp_colored_clust['auc']
        
        # Save virtual sensors contained in the cluster
        cluster_vs_df = cluster_suma_df[cluster_suma_df['auc'] != 0]
        for idx in cluster_vs_df.index:
            try:
                cluster_vs_df.loc[idx, 'virtual_sensor'] = suma2mne.loc[idx,'virtual_sensor_node']
            except:
                cluster_vs_df.drop(idx)
        
        # Write AUC per cluster vs to csv file
        cluster_vs_df = cluster_vs_df.drop_duplicates(subset='virtual_sensor').dropna().reset_index(drop=True)
        cluster_vs_df['virtual_sensor'] = cluster_vs_df['virtual_sensor'].astype(int)
        cluster_vs_df.to_csv(str(int(cluster))+'_'+hemi+'_virtual_sensor_auc.csv',index=None,sep=' ') 
        
        # Create gifti (SUMA Surface-based file) showing the clusters and AUC using SUMA vertices (NO INDEX)
        os.chdir(os.path.join(subj_freesurfer_dir,'SUMA'))
        cluster_suma_df.to_csv(os.path.join(subj_freesurfer_dir,'SUMA','auc_drop_ind.1D'),index=None,header=False,sep=' ')
        cmd="ConvertDset -o_gii -input auc_drop_ind.1D -prefix auc"
        subprocess.run(cmd,shell=True)
        
        #Create nifti (AFNI Volume-based file) showing the clusters and their AUC using SUMA vertices
        nii_outname='tmp_auc.nii'
        cluster_suma_df.to_csv(os.path.join(subj_freesurfer_dir,'SUMA','auc.1D.dset'), header=False,sep=' ')
        cmd="3dSurf2Vol -surf_A std.60.{}.smoothwm.gii -surf_B {}.pial.gii -sv sub-{}_{}_SurfVol.nii -spec std.60.sub-{}_{}_{}.spec -sdata_1D {} -grid_parent sub-{}_{}_SurfVol.nii -map_func nzave -f_steps 15 -prefix {}"
        cmd=cmd.format(hemi,hemi,SUBJECT,session,SUBJECT,session,hemi,'auc.1D.dset',SUBJECT,session,os.path.join(wdir,nii_outname))
        subprocess.run(cmd,shell=True)
        
        #Expand clusters into the volume
        os.chdir(wdir)
        cmd="@ROI_modal_grow -input {}/tmp_auc.nii -outdir {}/temp_grow -niters 3 -mask {}/tmp_auc.nii -prefix auc"
        cmd=cmd.format(wdir,wdir,wdir)
        subprocess.run(cmd,shell=True)
       
        # Unzip gifti
        for file in os.listdir('temp_grow'):
            if 'rm_rgm_03.nii.gz' in file:
                gunzip(os.path.join('temp_grow', file))
                os.rename(os.path.join('temp_grow',file[:-3]),os.path.join(wdir,'auc.nii'))  
        
        #Move cluster files into working directory
        shutil.move(os.path.join(subj_freesurfer_dir,'SUMA','auc.1D.dset'),os.path.join(wdir,str(int(cluster))+'_'+hemi+'_SUMA_auc.1D.dset'))
        shutil.move(os.path.join(subj_freesurfer_dir,'SUMA','auc.gii.dset'),os.path.join(wdir,str(int(cluster))+'_'+hemi+'_auc.gii'))
        shutil.move(os.path.join(wdir,'auc.nii'),os.path.join(wdir,str(int(cluster))+'_'+hemi+'_auc.nii'))
        
        #Remove temporary files
        shutil.rmtree(os.path.join(wdir,'temp_grow')) 
        os.remove(os.path.join(wdir,'tmp_auc.nii'))
        os.remove(os.path.join(subj_freesurfer_dir,'SUMA','auc_drop_ind.1D'))
    
    # Create gifti (SUMA Surface-based file) showing all clusters by color using SUMA vertices (NO INDEX)
    os.chdir(os.path.join(subj_freesurfer_dir,'SUMA'))
    colored_clust_df.to_csv(os.path.join(subj_freesurfer_dir,'SUMA','colored_clust_drop_ind.1D'),index=None,header=False,sep=' ')
    cmd="ConvertDset -o_gii -input colored_clust_drop_ind.1D -prefix colored_clusters"
    subprocess.run(cmd,shell=True)
    shutil.copyfile('colored_clusters.gii.dset', os.path.join(wdir, 'colored_clusters.gii'))
    
    #Prepare to merge cluster files into one gifti
    os.chdir(wdir)
    gifti_list = glob.glob('*{}*auc*gii*'.format(hemi))
    # Label gifti files
    if len(gifti_list) > 1:
        for file in gifti_list:
            if '1' in file:
                primary=file
            elif '2' in file:
                secondary=file
            elif '3' in file:
                tertiary=file
        
        # Merge the primary and secondary clusters
        cmd="3dcalc -a {} -b {} -expr '(step(a)*a)+(step(b)*b)' -prefix {}"
        cmd=cmd.format(primary, secondary,'{}_merged_clusters.gii'.format(hemi))
        subprocess.run(cmd,shell=True)
        
        # Merge combined file to tertiary cluster
        if len(clusters) > 2:
            shutil.move('{}_merged_clusters.gii'.format(hemi), 'temp_merge.gii')
            cmd="3dcalc -a {} -b {} -expr '(step(a)*a)+(step(b)*b)' -prefix {}"
            cmd=cmd.format('temp_merge.gii', tertiary,'{}_merged_clusters.gii'.format(hemi))
            subprocess.run(cmd,shell=True)
            os.remove('temp_merge.gii')
            
        # Copy back to SUMA folder for surface visualizations
        shutil.copyfile('{}_merged_clusters.gii'.format(hemi), os.path.join(subj_freesurfer_dir,'SUMA','{}_merged_clusters.gii'.format(hemi)))
        print("View merged surface files from within {}'s freesurfer SUMA folder.".format(SUBJECT))


