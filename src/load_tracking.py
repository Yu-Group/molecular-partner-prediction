import os
import sys
from copy import deepcopy
from os.path import join as oj

sys.path.append('..')
import mat4py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.external.tifffile import imread

pd.options.mode.chained_assignment = None  # default='warn' - caution: this turns off setting with copy warning
import pickle as pkl
from viz import *
import math
import config


def get_tracks(data_dir, split=None, all_data=False,
               processed_tracks_file=oj(config.DIR_TRACKS, 'tracks.pkl'),
               dset='orig'):
    '''Read and save tracks tracks from folders within data_dir into a dataframe
       Assumes (matlab) tracking has been run
    '''
    processed_tracks_file = processed_tracks_file[:-4] + '_' + dset + '.pkl'
    print('\t', processed_tracks_file, data_dir)

    if os.path.exists(processed_tracks_file):
        return pd.read_pickle(processed_tracks_file)
    dfs = []

    if split['feature_selection'] is None:
        split = None
    if split is not None:
        flatten = lambda l: [item for sublist in l for item in sublist]
        split = flatten(split.values())

    # 2 directories of naming
    for upper_dir in sorted(os.listdir(data_dir)):
        print('dirs', upper_dir)
        if upper_dir.startswith('.') or 'Icon' in upper_dir:
            continue
        for cell_dir in sorted(os.listdir(oj(data_dir, upper_dir))):
            
            print('\t', cell_dir)
            if not 'Cell' in cell_dir:
                continue
            cell_num = oj(upper_dir, cell_dir.replace('Cell', '').replace('_1s', ''))
            if split is not None:
                if not cell_num in split:
                    continue
            full_dir = f'{data_dir}/{upper_dir}/{cell_dir}'
            fname = full_dir + '/TagRFP/Tracking/ProcessedTracks.mat'
            print('\t', cell_num)
            
            # fname_image = oj(data_dir, upper_dir, cell_dir)
            mat = mat4py.loadmat(fname)
            tracks = mat['tracks']
            n = len(tracks['t'])
            
            
            # basic features
            t = np.array([tracks['t'][i] for i in range(n)])
            data = {
                'lifetime': tracks['lifetime_s'],                
                'cell_num': [cell_num] * n,                
                'catIdx': tracks['catIdx'],
                't': [t[i][0] for i in range(n)],
            }
            
            
            # displacement features
            totalDisplacement = []
            msd = []    # mean squared displacement
            for i in range(n):
                try:
                    totalDisplacement.append(tracks['MotionAnalysis'][i]['totalDisplacement'])
                except:
                    totalDisplacement.append(0)
                try:
                    msd.append(np.nanmax(tracks['MotionAnalysis'][i]['MSD']))
                except:
                    msd.append(0)
            data['mean_total_displacement'] = [totalDisplacement[i] / tracks['lifetime_s'][i] for i in range(n)]
            data['mean_square_displacement'] =  msd
            
            # position features
            x_pos_seq = np.array(
                [tracks['x'][i][0] for i in range(n)])  # x-position for clathrin (auxilin is very similar)
            y_pos_seq = np.array(
                [tracks['y'][i][0] for i in range(n)])  # y-position for clathrin (auxilin is very similar)
            data['x_pos_seq'] = x_pos_seq
            data['y_pos_seq'] = y_pos_seq
            data['x_pos'] = [sum(x) / len(x) for x in x_pos_seq]  # mean position in the image
            data['y_pos'] = [sum(y) / len(y) for y in y_pos_seq]
            
            # track features
            num_channels = len(tracks['A'][0])
            for idx_channel, prefix in zip(range(num_channels),
                                           ['X', 'Y', 'Z'][:num_channels]):
                track = np.array([tracks['A'][i][idx_channel] for i in range(n)])
                data[prefix + '_pvals'] = np.array([tracks['pval_Ar'][i][idx_channel] for i in range(n)])
                starts = []
                for d in tracks['startBuffer']:
                    if len(d) == 0:
                        starts.append([])
                    else:
                        starts.append(d['A'][idx_channel])
                ends = []
                for d in tracks['endBuffer']:
                    if len(d) == 0:
                        ends.append([])
                    else:
                        ends.append(d['A'][idx_channel])
                if prefix == 'X':
                    data[prefix + '_extended'] = [starts[i] + track[i] + ends[i] for i in range(n)]
                data[prefix] = track
                data[prefix + '_starts'] = starts
                data[prefix + '_ends'] = ends 
            data['lifetime_extended'] = [len(x) for x in data['X_extended']]

            # pixel features
            if all_data:
                cla, aux = get_images(full_dir)
                pixel = np.array([[cla[int(t[i][j]), int(y_pos_seq[i][j]), int(x_pos_seq[i][j])]
                                   if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                                  for i in range(n)])
                pixel_up = np.array(
                    [[cla[int(t[i][j]), min(int(y_pos_seq[i][j] + 1), cla.shape[1] - 1), int(x_pos_seq[i][j])]
                      if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                     for i in range(n)])
                pixel_down = np.array([[cla[int(t[i][j]), max(int(y_pos_seq[i][j] - 1), 0), int(x_pos_seq[i][j])]
                                        if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                                       for i in range(n)])
                pixel_left = np.array([[cla[int(t[i][j]), int(y_pos_seq[i][j]), max(int(x_pos_seq[i][j] - 1), 0)]
                                        if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                                       for i in range(n)])
                pixel_right = np.array(
                    [[cla[int(t[i][j]), int(y_pos_seq[i][j]), min(int(x_pos_seq[i][j] + 1), cla.shape[2] - 1)]
                      if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                     for i in range(n)])
                data['pixel'] = pixel
                data['pixel_left'] = pixel_left
                data['pixel_right'] = pixel_right
                data['pixel_up'] = pixel_up
                data['pixel_down'] = pixel_down
                data['center_max'] = [max(pixel[i]) for i in range(n)]
                data['left_max'] = [max(pixel_left[i]) for i in range(n)]
                data['right_max'] = [max(pixel_right[i]) for i in range(n)]
                data['up_max'] = [max(pixel_up[i]) for i in range(n)]
                data['down_max'] = [max(pixel_down[i]) for i in range(n)]
            
            df = pd.DataFrame.from_dict(data)
            dfs.append(deepcopy(df))
    df = pd.concat(dfs)
    os.makedirs(os.path.dirname(processed_tracks_file), exist_ok=True)
    df.to_pickle(processed_tracks_file)
    return df

def get_images(cell_dir: str):
    '''Loads in X and Y for one cell
    
    Params
    ------
    cell_dir
        Path to directory for one cell
    
    Returns
    -------
    X : np.ndarray
        has shape (W, H, num_images)
    Y : np.ndarray
        has shape (W, H, num_images)
    '''
    for name in os.listdir(oj(cell_dir, 'TagRFP')):
        if 'tif' in name:
            fname1 = name
    for name in os.listdir(oj(cell_dir, 'EGFP')):
        if 'tif' in name:
            fname2 = name
    X = imread(oj(cell_dir, 'TagRFP', fname1))  # .astype(np.float32) # X = RFP(clathrin) (num_images x H x W)
    Y = imread(oj(cell_dir, 'EGFP', fname2))  # .astype(np.float32) # Y = EGFP (auxilin) (num_image x H x W)
    return X, Y