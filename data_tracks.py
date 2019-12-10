from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from os.path import join as oj
from skimage.external.tifffile import imread
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
from style import *
import math
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import DictionaryLearning, NMF

# auxilin_dir = '/accounts/grad/xsli/auxilin_data'
auxilin_dir = '/scratch/users/vision/data/abc_data/auxilin_data_tracked'


def get_data(use_processed=True, save_processed=True, processed_file='processed/df.pkl',
             use_processed_dicts=True):
    '''
    Params
    ------
    use_processed: bool, optional
        determines whether to load df from cached pkl
    save_processed: bool, optional
        if not using processed, determines whether to save the df
    use_processed_dicts: bool, optional
        if False, recalculate the dictionary features
    '''
    if use_processed and os.path.exists(processed_file):
        return pd.read_pickle(processed_file)
    else:
        print('computing preprocessing...')
        df = get_tracks() # note: different Xs can be different shapes
        df = remove_invalid_tracks(df)
        df = preprocess(df)
        df = add_outcomes(df)
        df = remove_tracks_by_lifetime(df, outcome_key='y', plot=False, acc_thresh=0.90)
        df = add_dict_features(df, use_processed=use_processed_dicts)
        df = add_smoothed_tracks(df)
        if save_processed:
            df.to_pickle(processed_file)
    return df

def get_images(cell_name, auxilin_dir=auxilin_dir):
    '''Loads in X and Y for one cell
    
    Returns
    -------
    X : np.ndarray
        has shape (W, H, num_images)
    Y : np.ndarray
        has shape (W, H, num_images)
    '''
    #cell_name = 'Cell1_1s'
    data_dir = oj(auxilin_dir, 'A7D2', cell_name) # 'A7D2', 'EGFB-GAK-F6'
    for name in os.listdir(oj(data_dir, 'TagRFP')):
        if 'tif' in name:
            fname1 = name
    for name in os.listdir(oj(data_dir, 'EGFP')):
        if 'tif' in name:
            fname2 = name
    X = imread(oj(data_dir, 'TagRFP', fname1)) #.astype(np.float32) # X = RFP(clathrin) (num_images x H x W)
    Y = imread(oj(data_dir, 'EGFP', fname2)) #.astype(np.float32) # Y = EGFP (auxilin) (num_image x H x W)  
    return X, Y

def get_tracks(cell_nums=[1, 2, 3, 4, 5, 6], all_data=False):
    dfs = []
    # 8 cell folders [1, 2, 3, ..., 8]
    for cell_num in cell_nums:
        fname = f'{auxilin_dir}/A7D2/Cell{cell_num}_1s/TagRFP/Tracking/ProcessedTracks.mat'
        cla, aux = get_images(f'Cell{cell_num}_1s', auxilin_dir=auxilin_dir)
        fname_image = data_dir = oj(auxilin_dir, 'A7D2', f'Cell{cell_num}_1s')
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
        t = np.array([tracks['t'][i] for i in range(n)])
        x_pos_seq = np.array([tracks['x'][i][0] for i in range(n)]) # position for clathrin (auxilin is very similar)
        y_pos_seq = np.array([tracks['y'][i][0] for i in range(n)])
        pixel = np.array([[cla[int(t[i][j]), int(y_pos_seq[i][j]), int(x_pos_seq[i][j])]
                              if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                             for i in range(n)])
        pixel_up = np.array([[cla[int(t[i][j]), min(int(y_pos_seq[i][j] + 1), cla.shape[1] - 1), int(x_pos_seq[i][j])]
                              if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                             for i in range(n)])
        pixel_down = np.array([[cla[int(t[i][j]), max(int(y_pos_seq[i][j] - 1), 0), int(x_pos_seq[i][j])]
                              if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                             for i in range(n)])
        pixel_left = np.array([[cla[int(t[i][j]), int(y_pos_seq[i][j]), max(int(x_pos_seq[i][j]-1), 0)]
                              if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                             for i in range(n)])
        pixel_right = np.array([[cla[int(t[i][j]), int(y_pos_seq[i][j]), min(int(x_pos_seq[i][j]+1), cla.shape[2]-1)]
                              if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                             for i in range(n)])                            
        X_pvals = np.array([tracks['pval_Ar'][i][0] for i in range(n)])
        Y_pvals = np.array([tracks['pval_Ar'][i][1] for i in range(n)])
    #     df = pd.DataFrame(tracks)
    #     print(df.keys()) # these lines help us look at the other stored vars
        data = {
            'X': X, 
            'Y': Y,
            'X_pvals': X_pvals,
            'Y_pvals': Y_pvals,
            'pixel': pixel,
            'pixel_left': pixel_left,
            'pixel_right': pixel_right,
            'pixel_up': pixel_up,
            'pixel_down': pixel_down,
            'catIdx': tracks['catIdx'],
            'total_displacement': totalDisplacement,
            'mean_square_displacement': msd,
            'lifetime': tracks['lifetime_s'],
            'x_pos': [sum(x) / len(x) for x in x_pos_seq], # mean position in the image
            'y_pos': [sum(y) / len(y) for y in y_pos_seq],
            'center_max': [max(pixel[i]) for i in range(n)],
            'left_max': [max(pixel_left[i]) for i in range(n)],
            'right_max': [max(pixel_right[i]) for i in range(n)],
            'up_max': [max(pixel_up[i]) for i in range(n)],
            'down_max': [max(pixel_down[i]) for i in range(n)],
            'cell_num': [cell_num] * n,
        }
        if all_data:
            data['x_pos_seq'] = x_pos_seq
            data['y_pos_seq'] = y_pos_seq
        df = pd.DataFrame.from_dict(data)
        dfs.append(deepcopy(df))
    return pd.concat(dfs)

def preprocess(df):
    '''Add a bunch of extra features to the df
    '''
    df = df[df.lifetime > 2]
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

def add_outcomes(df, thresh=3.25, p_thresh=0.05, aux_peak=642.3754691658837):
    '''Add binary outcome of whether spike happened and info on whether events were questionable
    '''
    df['y_score'] = df['Y_max'].values - (df['Y_mean'].values + thresh * df['Y_std'].values)
    df['y_thresh'] = (df['y_score'].values > 0).astype(np.int) # Y_max was big
    df['y'] = df['Y_max'] > aux_peak
    
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


def extract_X_mat(df):
    '''Extract matrix for X filled with zeros after sequences
    Width of matrix is length of longest lifetime
    '''
    p = df.lifetime.max()
    n = df.shape[0]
    X_mat = np.zeros((n, p)).astype(np.float32)
    X = df['X'].values
    for i in range(n):
        x = X[i]
        num_timepoints = min(p, len(x))
        X_mat[i, :num_timepoints] = x[:num_timepoints]
    X_mat = np.nan_to_num(X_mat)
    X_mat -= np.min(X_mat)
    X_mat /= np.std(X_mat)
    return X_mat


def remove_tracks_by_lifetime(df, outcome_key='y_thresh', plot=False, acc_thresh=0.95):
    '''Calculate accuracy you can get by just predicting max class 
    as a func of lifetime and return points within proper lifetime
    '''
    vals = df[['lifetime', outcome_key]]
    R, C = 1, 3
    lifetimes = np.unique(vals['lifetime'])

    props1 = np.array([1 - np.mean(vals[outcome_key][vals['lifetime'] <= l]) for l in lifetimes])
    idx_thresh = np.nonzero(props1 <= acc_thresh)[0][0]
    thresh_lower = lifetimes[idx_thresh]
    n = df.shape[0]    
    
    props2 = np.array([np.mean(vals[outcome_key][vals['lifetime'] >= l]) for l in lifetimes])
    idx_thresh_2 = np.nonzero(props2 >= acc_thresh)[0][0]
    thresh_higher = lifetimes[idx_thresh_2]    

    if plot:
        plt.figure(figsize=(12, 4), dpi=200)
        plt.subplot(R, C, 1)
        outcome = df[outcome_key]
        plt.hist(df['lifetime'][outcome==1], label='aux+', alpha=1, color=cb, bins=25)
        plt.hist(df['lifetime'][outcome==0], label='aux-', alpha=0.7, color=cr, bins=25)
        plt.xlabel('lifetime')
        plt.ylabel('count')
        plt.legend()
    
        plt.subplot(R, C, 2)
        plt.plot(lifetimes, props1, color=cr)
    #     plt.axvline(thresh_lower)
        plt.axvspan(0, thresh_lower, alpha=0.2, color=cr)
        plt.ylabel('fraction of negative events')
        plt.xlabel(f'lifetime <= value\nshaded includes {np.sum(vals["lifetime"]<=thresh_lower)/n*100:0.0f}% of pts')


        plt.subplot(R, C, 3)
        plt.plot(lifetimes, props2, cb)
        plt.axvspan(thresh_higher, max(lifetimes), alpha=0.2, color=cb)
        plt.ylabel('fraction of positive events')
        plt.xlabel(f'lifetime >= value\nshaded includes {np.sum(vals["lifetime"]>=thresh_higher)/n*100:0.0f}% of pts')
        plt.tight_layout()
        plt.show()

    # only df with lifetimes in proper range
    df = df[(df['lifetime'] > thresh_lower) & (df['lifetime'] < thresh_higher)]
    
    return df

def add_dict_features(df, sc_comps_file='processed/dictionaries/sc_12_alpha=1.pkl', 
                      nmf_comps_file='processed/dictionaries/nmf_12.pkl', 
                      use_processed=True):
    '''Add features from saved dictionary to df
    '''
    def sparse_code(X_mat, n_comps=12, alpha=1, out_dir='processed/dictionaries'):
        print('sparse coding...')
        d = DictionaryLearning(n_components=n_comps, alpha=alpha)
        d.fit(X_mat)
        pkl.dump(d, open(oj(out_dir, f'sc_{n_comps}_alpha={alpha}.pkl'), 'wb'))

    def nmf(X_mat, n_comps=12, out_dir='processed/dictionaries'):
        print('running nmf...')
        d = NMF(n_components=n_comps)
        d.fit(X_mat)
        pkl.dump(d, open(oj(out_dir, f'nmf_{n_comps}.pkl'), 'wb'))
    
    X_mat = extract_X_mat(df)
    X_mat -= np.min(X_mat)
    
    # if feats don't exist, compute them
    if not use_processed or not os.path.exists(sc_comps_file):
        os.makedirs('processed/dictionaries', exist_ok=True)
        sparse_code(X_mat)
        nmf(X_mat)
    
    try:
        # sc
        d_sc = pkl.load(open(sc_comps_file, 'rb'))
        encoding = d_sc.transform(X_mat)
        for i in range(encoding.shape[1]):
            df[f'sc_{i}'] = encoding[:, i]

        # nmf
        d_nmf = pkl.load(open(nmf_comps_file, 'rb'))
        encoding_nmf = d_nmf.transform(X_mat)
        for i in range(encoding_nmf.shape[1]):
            df[f'nmf_{i}'] = encoding_nmf[:, i]
    except:
        print('dict features not added!')
    return df
                   
def add_smoothed_tracks(df, 
                        method='spline', 
                        s_spl=0.004):
    X_smooth_spl = []
    X_smooth_spl_dx = []
    X_smooth_spl_d2x = []
    def num_local_maxima(x):
        return(len([i for i in range(1, len(x)-1) if x[i] > x[i-1] and x[i] > x[i+1]]))
    for x in df['X']:
        spl = UnivariateSpline(x=range(len(x)), 
                               y=x, 
                               w=[1.0/len(x)]*len(x),
                               s=np.var(x)*s_spl)                   
        spl_dx = spl.derivative()
        spl_d2x = spl_dx.derivative()
        X_smooth_spl.append(spl(range(len(x))))
        X_smooth_spl_dx.append(spl_dx(range(len(x))))
        X_smooth_spl_d2x.append(spl_d2x(range(len(x))))
    df['X_smooth_spl'] = np.array(X_smooth_spl)
    df['X_smooth_spl_dx'] = np.array(X_smooth_spl_dx)
    df['X_smooth_spl_d2x'] = np.array(X_smooth_spl_d2x)
    df['X_max_spl'] = np.array([np.max(x) for x in X_smooth_spl])
    df['dx_max_spl'] = np.array([np.max(x) for x in X_smooth_spl_dx])
    df['d2x_max_spl'] = np.array([np.max(x) for x in X_smooth_spl_d2x])               
    df['num_local_max_spl'] = np.array([num_local_maxima(x) for x in X_smooth_spl])
    df['num_local_min_spl'] = np.array([num_local_maxima(-1 * x) for x in X_smooth_spl])
    return df
                   