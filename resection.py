#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Mon Oct 21 16:03:45 2024
@author: Leela Srinivasan

"""

import os
import glob
import shutil
import subprocess
import pandas as pd


from general import euclidean_distance


def import_resection(config):
    """
    

    Parameters
    ----------
    config : dict
        configuration dictionary describing user paths.

    Returns
    -------
    None.

    """
    
    if not os.path.isfile('rsxn_al.nii'):
        shutil.copyfile((os.path.join(config['resection_dir'],'rsxn.msk.nii')),'rsxn.msk.nii')
        
        
def align_resection_to_surfvol(SUBJECT, SESSION, f):
    """
    
    Align the resection mask to the SurfVol by translating the relationship between the preoperative t1 and the SurfVol
    
    Parameters
    ----------
    SUBJECT : string
        p***.
    SESSION : string
        clinical/altclinical.
    f : string
        resection mask filename.

    Returns
    -------
    None. Outputs aligned resection to the working directory.

    """
    
    if not os.path.exists('rsxn_al.nii'):
        
        cmd1="3dAllineate -base sub-{}_{}_SurfVol.nii -source resection_t1.nii -prefix aligned+orig -1Dmatrix_save anat_to_fs"
        cmd1=cmd1.format(SUBJECT,SESSION)
        subprocess.run(cmd1,shell=True)
        
        cmd2="3dAllineate -base sub-{}_{}_SurfVol.nii -source rsxn.msk.nii -1Dmatrix_apply anat_to_fs.aff12.1D -prefix tmp_rsxn+orig"
        cmd2=cmd2.format(SUBJECT,SESSION)
        subprocess.run(cmd2,shell=True)
        
        cmd3="3dcalc -a tmp_rsxn+orig -expr 'ispositive(a-0.1)' -prefix tmp2_rsxn.nii"
        subprocess.run(cmd3,shell=True)
        
        cmd4="3dmask_tool -input tmp2_rsxn.nii -prefix rsxn_al.nii -fill_holes"
        subprocess.run(cmd4,shell=True)
        
        for file in os.listdir(os.getcwd()):
            if 'tmp_rsxn+orig' in file or 'tmp2_rsxn' in file:
                os.remove(file)
                

def get_resection_cm(f):
    """
    

    Parameters
    ----------
    f : string
        filename of aligned resection mask.

    Returns
    -------
    rsxn_cm : int
        Center of mass of the resection mask

    """
    
    #Calculate CM
    cmd1="3dcm {}"
    cmd1=cmd1.format(f)
    result=subprocess.run(cmd1,shell=True,capture_output=True,text = True)
    
    #Read and return CM
    rsxn_cm_str = result.stdout.split('\n')[0].split('  ')
    rsxn_cm = [eval(coord) for coord in rsxn_cm_str]
    return rsxn_cm
                

def get_resection_cluster_overlap(SUBJECT, rsxn_cm):
    """
    

    Parameters
    ----------
    SUBJECT : string
        p***.
    rsxn_cm : int
        Center of mass of the resection mask

    Returns
    -------
    output_list : array
        array of 3-element arrays containing (1) cluster number/hemisphere, (2) percentage overlap with the resection, 
        and (3) the distance between the COM of the cluster and the resection.
        
    """
    #Find cluster files projected into the volume
    output_list = []
    for nifti in glob.glob("*_auc.nii"):
        
        # Get file prefix
        f_pref = nifti.split('_auc')[0]
    
        # Pipe overlap to text file
        cmd1="3dABoverlap rsxn_al.nii {} > overlap.txt"
        cmd1=cmd1.format(nifti)
        subprocess.run(cmd1,shell=True)
    
        #Percentage of the resection contained in the cluster
        overlap_df = pd.read_csv('overlap.txt', sep='\s+', skiprows=1)
        percentage_overlap = (overlap_df.values[0][3]/overlap_df.values[0][0])*100
        print('Overlap for subject {}, {}: {}'.format(SUBJECT, f_pref, percentage_overlap))

        # Get cluster COM
        cmd2="3dcm {}"
        cmd2=cmd2.format(nifti)
        result=subprocess.run(cmd2,shell=True,capture_output=True,text = True)
        cluster_cm_str = result.stdout.split('\n')[0].split('  ')
        cluster_cm = [eval(coord) for coord in cluster_cm_str]
        
        # Calculate euclidean distance between COMs
        com_dist = euclidean_distance(cluster_cm, rsxn_cm)
        print('Distance from {} resection COM to {} COM: {}'.format(SUBJECT, f_pref, com_dist))

        output_list.append([f_pref, percentage_overlap, com_dist])
    
    return output_list


def print_overlap_results(output_list):
    """
    

    Parameters
    ----------
    output_list : array
        list of lists from get_resection_cluster_overlap with overlap information per cluster

    Returns
    -------
    None.

    """

    for cluster in output_list:
        print('{} has overlap percentage {} and has inter-CM distance of {}'.format(cluster[0], cluster[1], cluster[2]))
    
    
    
