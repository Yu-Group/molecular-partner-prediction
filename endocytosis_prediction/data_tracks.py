from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from os.path import join as oj
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.decomposition import sparse_encode
import numpy as np
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F
import torch
from copy import deepcopy
from sklearn import metrics
plt.style.use('dark_background')
import mat4py
import pandas as pd
import pickle as pkl

def get_data():
    df = get_tracks() # note: different Xs can be different shapes
    df = remove_invalid_tracks(df)
    df = preprocess(df)
    df = add_outcomes(df)
    df = add_sparse_coding_features(df)
    return df

def get_tracks(cell_nums=[1, 2, 3, 4, 5], all_data=False):
    dfs = []
    # 8 cell folders [1, 2, 3, ..., 8]
    for cell_num in cell_nums:
        fname = f'/scratch/users/vision/data/abc_data/auxilin_data_tracked/A7D2/Cell{cell_num}_1s/TagRFP/Tracking/ProcessedTracks.mat'
        mat = mat4py.loadmat(fname)
        tracks = mat['tracks']
        n = len(tracks['t'])
        totalDisplacement = []
        msd = [] # mean squared displacement
        for i in range(n):
            try:
                totalDisplacement.append(tracks['MotionAnalysis'][i]['totalDisplacement'])
            except:
                totalDisplacement.append(0)
            try:
                msd.append(np.nanmax(tracks['MotionAnalysis'][i]['MSD']))
            except:
                msd.append(0)
        X = np.array([tracks['A'][i][0] for i in range(n)])
        Y = np.array([tracks['A'][i][1] for i in range(n)])
        x_pos_seq = np.array([tracks['x'][i][0] for i in range(n)]) # position for clathrin (auxilin is very similar)
        y_pos_seq = np.array([tracks['y'][i][0] for i in range(n)])
        X_pvals = np.array([tracks['pval_Ar'][i][0] for i in range(n)])
        Y_pvals = np.array([tracks['pval_Ar'][i][1] for i in range(n)])
    #     df = pd.DataFrame(tracks)
    #     print(df.keys()) # these lines help us look at the other stored vars
        data = {
            'X': X, 
            'Y': Y,
            'X_pval': X_pvals,
            'Y_pvals': Y_pvals,
            'catIdx': tracks['catIdx'],
            'total_displacement': totalDisplacement,
            'mean_square_displacement': msd,
            'lifetime': tracks['lifetime_s'],
            'x_pos': [sum(x) / len(x) for x in x_pos_seq], # mean position in the image
            'y_pos': [sum(y) / len(y) for y in y_pos_seq],
            'cell_num': [cell_num] * n,
        }
        if all_data:
            data['x_pos_seq'] = x_pos_seq
            data['y_pos_seq'] = y_pos_seq
        df = pd.DataFrame.from_dict(data)
        df['len'] = np.array([len(x) - np.sum(np.isnan(x)) for x in df.X.values])
        dfs.append(deepcopy(df))
    return pd.concat(dfs)

def preprocess(df):
    '''Add a bunch of extra features to the df
    '''
    df = df[df.len > 2]
    df['X_max'] = np.array([max(x) for x in df.X.values])
    df['X_min'] = np.array([max(x) for x in df.X.values])
    df['X_mean'] = np.nan_to_num(np.array([np.nanmean(x) for x in df.X.values]))
    df['X_std'] = np.nan_to_num(np.array([np.std(x) for x in df.X.values]))
    df['Y_max'] = np.array([max(y) for y in df.Y.values])    
    df['Y_mean'] = np.nan_to_num(np.array([np.nanmean(y) for y in df.Y.values]))
    df['Y_std'] = np.nan_to_num(np.array([np.std(y) for y in df.Y.values]))
    
    # hand-engineeredd features
    def calc_rise(x):
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        rise = val_max - np.min(x[:idx_max + 1]) # max change before peak
        return rise

    def calc_fall(x):
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        fall = val_max - np.min(x[idx_max:]) # drop after peak
        return fall
    
    def max_diff(x): return np.max(np.diff(x))
    def min_diff(x): return np.min(np.diff(x))
    
    df['rise'] = df.apply(lambda row: calc_rise(row['X']), axis=1)
    df['fall'] = df.apply(lambda row: calc_fall(row['X']), axis=1)
    df['max_diff'] = df.apply(lambda row: max_diff(row['X']), axis=1)    
    df['min_diff'] = df.apply(lambda row: min_diff(row['X']), axis=1)        
    return df

def add_outcomes(df, thresh=3.25, p_thresh=0.05):
    '''Add binary outcome of whether spike happened and info on whether events were questionable
    '''
    df['y_score'] = df['Y_max'].values - (df['Y_mean'].values + thresh * df['Y_std'].values)
    df['y_thresh'] = (df['y_score'].values > 0).astype(np.int) # Y_max was big
    
    # outcomes based on significant p-values
    num_sigs = [np.array(df['Y_pvals'].iloc[i]) < p_thresh for i in range(df.shape[0])]
    df['y_single_sig'] = np.array([num_sigs[i].sum() > 0 for i in range(df.shape[0])]).astype(np.int)
    df['y_double_sig'] = np.array([num_sigs[i].sum() > 1 for i in range(df.shape[0])]).astype(np.int)
    y_consec_sig = []
    for i in range(df.shape[0]):
        idxs_sig = np.where(num_sigs[i]==1)[0] # indices of significance
        
        # find whether there were consecutive sig. indices
        if len(idxs_sig) > 1 and np.min(np.diff(idxs_sig)) == 1:
            y_consec_sig.append(1)
        else:
            y_consec_sig.append(0)
    df['y_consec_sig'] = y_consec_sig
    return df

def remove_invalid_tracks(df, keep=[1, 2]):
    '''Remove certain types of tracks based on cat_idx
    cat_idx (idx 1 and 2)
    1-4 (non-complex trajectory - no merges and splits)
        1 - valid
        2 - signal occasionally drops out
        3 - cut  - starts / ends
        4 - multiple - at the same place (continues throughout)
    5-8 (there is merging or splitting)
    '''
    return df[df.catIdx.isin(keep)]


def extract_X_mat(df, p=300):
    '''Extract matrix for X filled with zeros after sequences end
    '''
    n = df.shape[0]
    X_mat = np.zeros((n, p)).astype(np.float32)
    X = df['X'].values
    for i in range(n):
        x = X[i]
        num_timepoints = min(300, len(x))
        X_mat[i, :num_timepoints] = x[:num_timepoints]
    X_mat = np.nan_to_num(X_mat)
    X_mat -= np.min(X_mat)
    X_mat /= np.std(X_mat)
    return X_mat

def add_sparse_coding_features(df, comps_file='comps_12_alpha=1.pkl'):
    '''Add features from saved dictionary to df
    '''
    X_mat = extract_X_mat(df)
    comps = pkl.load(open(comps_file, 'rb'))
    encoding = sparse_encode(X_mat, comps)
    for i in range(encoding.shape[1]):
        df[f'sc_{i}'] = encoding[:, i]
    return df