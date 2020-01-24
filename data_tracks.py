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
pd.options.mode.chained_assignment = None  # default='warn' - caution: this turns off setting with copy warning
import pickle as pkl
from style import *
import math
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import DictionaryLearning, NMF
from sklearn import decomposition


# auxilin_dir = '/accounts/grad/xsli/auxilin_data'
auxilin_dir = '/scratch/users/vision/data/abc_data/auxilin_data_tracked'

# data splitting
cell_nums_feature_selection = np.array([1])
cell_nums_train = np.array([1, 2, 3, 4, 5])
cell_nums_test = np.array([6])



def get_data(use_processed=True, save_processed=True, 
             processed_file='processed/df.pkl', metadata_file='processed/metadata.pkl',
             use_processed_dicts=True, outcome_def='y_consec_thresh', remove_hotspots=True, 
             frac_early=0, frac_late=0.15):
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
        print('loading + preprocessing data...')
        metadata = {}
        print('\tloading tracks...')
        df = get_tracks() # note: different Xs can be different shapes
        df['pid'] = np.arange(df.shape[0]) # assign each track a unique id
        metadata['num_tracks_orig'] = df.shape[0]
        
        print('\tpreprocessing data...')
        df = remove_invalid_tracks(df)
        df = preprocess(df)
        df = add_outcomes(df)
        metadata['num_tracks_valid'] = df.shape[0]
        metadata['num_aux_pos_valid'] = df[outcome_def].sum()
        metadata['num_hospots_valid'] = df['hotspots'].sum()

        if remove_hotspots:
            print('\tremoving hotspots....')
            df = df[df['hotspots']==0]
        metadata['num_tracks_after_hotspots'] = df.shape[0]
        metadata['num_aux_pos_after_hotspots'] = df[outcome_def].sum()
        
        '''
        df, meta_peaks = remove_tracks_by_peak_time(df, outcome_def, frac_early=frac_early, frac_late=frac_late)
        metadata.update(meta_peaks)
        '''
        
        df, meta_lifetime = remove_tracks_by_lifetime(df, outcome_def=outcome_def, plot=False, acc_thresh=0.92)
        metadata.update(meta_lifetime)
        
        
        print('\tadding features...')
        # df = add_dict_features(df, use_processed=use_processed_dicts)
        df = add_smoothed_tracks(df)
        df = add_pcs(df)
        if save_processed:
            pkl.dump(metadata, open(metadata_file, 'wb'))
            df.to_pickle(processed_file)
    return df

# gt labels (by pid)
def get_labels():
    return {
        'hotspots': [6510, 6606, 2373, 6135, 6023, 7730, 2193, 8307, 5626, 4109, 2921, 4614, 2573, 7490, 6097, 
            7836, 1011, 6493, 5779, 8660, 6232, 6009, 2579, 929, 3824, 357, 6162, 477, 5640, 6467, 
            244, 2922, 4288, 2926, 1480, 4441, 4683],
        'neg': [6725, 909, 5926, 983, 8224, 3363],
        'pos': [3982, 8243, 777, 3940, 7559, 2455, 4748, 633, 2177, 1205, 603, 7972, 8458, 3041, 924, 8786, 4116, 885, 6298, 4658, 7889, 982, 829, 1210, 3054, 504, 1164, 347, 627, 1470, 2662, 2813, 422, 8400, 7474, 1273, 6365, 1559, 4348, 1156, 6250, 4864, 639, 930, 5424, 7818, 8463, 4358, 7656, 843, 890, 4373, 2737, 7524, 2590, 3804, 7667, 2148, 8585, 2919, 5712, 3331, 4440, 1440, 4699, 4803, 1089, 3004, 3126, 2869, 4183, 7335, 3166, 8461, 2180, 849, 6458, 4575, 4091, 3966, 4725, 2514, 7626, 3055, 4200, 6429, 1220, 4472, 8559, 412, 903, 5440, 1084, 2136, 6833, 1189, 7521, 8141, 7939, 8421, 944, 1264, 298, 6600, 1309, 3043, 243, 4161, 6813, 5464]
    }

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

def get_tracks(cell_nums=[1, 2, 3, 4, 5, 6], all_data=False, processed_tracks_file='processed/tracks.pkl'):
    if os.path.exists(processed_tracks_file):
        return pd.read_pickle(processed_tracks_file)
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
    df = pd.concat(dfs)
    os.makedirs(os.path.dirname(processed_tracks_file), exist_ok=True)
    df.to_pickle(processed_tracks_file)
    return df

def preprocess(df):
    '''Add a bunch of extra features to the df
    '''
    df = df[df.lifetime > 2]
    df['X_max'] = np.array([max(x) for x in df.X.values])
    df['X_min'] = np.array([min(x) for x in df.X.values])
    df['X_mean'] = np.nan_to_num(np.array([np.nanmean(x) for x in df.X.values]))
    df['X_std'] = np.nan_to_num(np.array([np.std(x) for x in df.X.values]))
    df['Y_max'] = np.array([max(y) for y in df.Y.values])    
    df['Y_mean'] = np.nan_to_num(np.array([np.nanmean(y) for y in df.Y.values]))
    df['Y_std'] = np.nan_to_num(np.array([np.std(y) for y in df.Y.values]))
    df['X_peak_idx'] = np.nan_to_num(np.array([np.argmax(x) for x in df.X]))
    df['Y_peak_idx'] = np.nan_to_num(np.array([np.argmax(y) for y in df.Y]))
    df['X_peak_time_frac'] = df['X_peak_idx'].values / df['lifetime'].values
    df['slope_end'] = df.apply(lambda row: (row['X_max'] - row['X'][-1]) / (row['lifetime'] - row['X_peak_idx']), axis=1)
    df['X_peak_last_15'] = df['X_peak_time_frac'] >= 0.85
    df['X_peak_last_5'] = df['X_peak_time_frac'] >= 0.95
    
    # hand-engineeredd features
    def calc_rise(x):
        '''max change before peak
        '''
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        return val_max - np.min(x[:idx_max + 1])

    def calc_fall(x):
        '''max change after peak
        '''
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        return val_max - np.min(x[idx_max:])
    
    def calc_rise_slope(x):
        '''slope to max change before peak
        '''
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        x_early = x[:idx_max + 1]
        idx_min = np.argmin(x_early)
        denom = (idx_max - idx_min)
        if denom == 0:
            return 0
        return (val_max - np.min(x_early)) / denom


    def calc_fall_slope(x):
        '''slope to max change after peak
        '''
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        x_late = x[idx_max :]
        idx_min = np.argmin(x_late)
        denom = idx_min
        if denom == 0:
            return 0
        return (val_max - np.min(x_late)) / denom
    
    
    def max_diff(x): return np.max(np.diff(x))
    def min_diff(x): return np.min(np.diff(x))
    
    df['rise'] = df.apply(lambda row: calc_rise(row['X']), axis=1)
    df['fall'] = df.apply(lambda row: calc_fall(row['X']), axis=1)
    df['rise_slope'] = df.apply(lambda row: calc_rise_slope(row['X']), axis=1)
    df['fall_slope'] = df.apply(lambda row: calc_fall_slope(row['X']), axis=1)
    num = 3
    df['rise_local_3'] = df.apply(lambda row: 
                                  calc_rise(np.array(row['X'][max(0, row['X_peak_idx'] - num): 
                                                            row['X_peak_idx'] + num + 1])), 
                                          axis=1)
    df['fall_local_3'] = df.apply(lambda row: 
                                  calc_fall(np.array(row['X'][max(0, row['X_peak_idx'] - num): 
                                                            row['X_peak_idx'] + num + 1])), 
                                          axis=1)
    
    num2 = 11
    df['rise_local_11'] = df.apply(lambda row: 
                                   calc_rise(np.array(row['X'][max(0, row['X_peak_idx'] - num2): 
                                                            row['X_peak_idx'] + num2 + 1])), 
                                          axis=1)
    df['fall_local_11'] = df.apply(lambda row: 
                                   calc_fall(np.array(row['X'][max(0, row['X_peak_idx'] - num2): 
                                                            row['X_peak_idx'] + num2 + 1])), 
                                          axis=1)
    df['max_diff'] = df.apply(lambda row: max_diff(row['X']), axis=1)    
    df['min_diff'] = df.apply(lambda row: min_diff(row['X']), axis=1)        
    return df

def add_outcomes(df, thresh=3.25, p_thresh=0.05, aux_peak=642.375, aux_thresh=973):
    '''Add binary outcome of whether spike happened and info on whether events were questionable
    '''
    df['y_score'] = df['Y_max'].values - (df['Y_mean'].values + thresh * df['Y_std'].values)
    df['y_thresh'] = (df['y_score'].values > 0).astype(np.int) # Y_max was big
    df['y'] = df['Y_max'] > aux_peak
    
    # outcomes based on significant p-values
    num_sigs = [np.array(df['Y_pvals'].iloc[i]) < p_thresh for i in range(df.shape[0])]
    df['y_single_sig'] = np.array([num_sigs[i].sum() > 0 for i in range(df.shape[0])]).astype(np.int)
    df['y_double_sig'] = np.array([num_sigs[i].sum() > 1 for i in range(df.shape[0])]).astype(np.int)
    df['y_conservative_thresh'] = (df['Y_max'].values > aux_thresh).astype(np.int)
    y_consec_sig = []
    for i in range(df.shape[0]):
        idxs_sig = np.where(num_sigs[i]==1)[0] # indices of significance
        
        # find whether there were consecutive sig. indices
        if len(idxs_sig) > 1 and np.min(np.diff(idxs_sig)) == 1:
            y_consec_sig.append(1)
        else:
            y_consec_sig.append(0)
    df['y_consec_sig'] = y_consec_sig
    df['y_consec_thresh'] = np.logical_or(df['y_consec_sig'], df['y_conservative_thresh'])
    df['y_consec_thresh'][df.pid.isin(get_labels()['pos'])] = 1 # add manual pos labels
    df['y_consec_thresh'][df.pid.isin(get_labels()['neg'])] = 0 # add manual neg labels    
    
    def add_hotspots(df, num_sigs, outcome_def='consec_sig'):
        '''Identify hotspots as any track which over its time course has multiple events
        events must meet the event definition, then for a time not meet it, then meet it again
        Example: two consecutive significant p-values, then non-significant p-value, then 2 more consecutive p-values
        '''

        if outcome_def=='consec_sig':
            hotspots = np.zeros(df.shape[0]).astype(np.int)
            for i in range(df.shape[0]):
                idxs_sig = np.where(num_sigs[i]==1)[0] # indices of significance
                if idxs_sig.size < 5:
                    hotspots[i] = 0
                else:
                    diffs = np.diff(idxs_sig)
                    consecs = np.where(diffs==1)[0] # diffs==1 means there were consecutive sigs
                    consec_diffs = np.diff(consecs)
                    if consec_diffs.shape[0] > 0 and np.max(consec_diffs) > 2: # there were greated than 2 non-consec sigs between the consec sigs
                        hotspots[i] = 1
                    else:
                        hotspots[i] = 0
        df['sig_idxs'] = num_sigs
        df['hotspots'] = hotspots
        df['hotspots'][df.pid.isin(get_labels()['hotspots'])] = 1 # add manual hotspot labels
        return df
    
    df = add_hotspots(df, num_sigs)
    
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

def remove_tracks_by_peak_time(df: pd.DataFrame, outcome_def, remove_key='X_peak_idx', frac_early=0, frac_late=0.15):
    '''Remove tracks where aux peaks in beginning / end
    '''
    early_peaks = df[remove_key] < df['lifetime'] * frac_early
    late_peaks = df[remove_key] > (df['lifetime'] * (1 - frac_late))
    
    df2 = df[np.logical_and(~early_peaks, ~late_peaks)]
    meta = {
        'num_peaks_early': early_peaks.sum(),
        'num_aux_pos_early': df[outcome_def].values[early_peaks].sum(),
        'num_peaks_late': late_peaks.sum(),
        'num_aux_pos_late': df[outcome_def].values[late_peaks].sum(),
        'num_tracks_after_peak_time': df2.shape[0],
        'num_aux_pos_after_peak_time': df2[outcome_def].sum()
    }
    return df2, meta

def remove_tracks_by_lifetime(df: pd.DataFrame, outcome_def: str, plot=False, acc_thresh=0.95):
    '''Calculate accuracy you can get by just predicting max class 
    as a func of lifetime and return points within proper lifetime
    '''
    vals = df[['lifetime', outcome_def]]
    R, C = 1, 3
    lifetimes = np.unique(vals['lifetime'])

    # cumulative accuracy for different thresholds
    accs_cum_lower = np.array([1 - np.mean(vals[outcome_def][vals['lifetime'] <= l]) for l in lifetimes])
    idx_thresh = np.nonzero(accs_cum_lower >= acc_thresh)[0][-1] # last nonzero index
    thresh_lower = lifetimes[idx_thresh]    
    
    accs_cum_higher = np.array([np.mean(vals[outcome_def][vals['lifetime'] >= l]) for l in lifetimes]).flatten()
    try:
        idx_thresh_2 = np.nonzero(accs_cum_higher >= acc_thresh)[0][0]
        thresh_higher = lifetimes[idx_thresh_2]
    except:
        idx_thresh_2 = lifetimes.size - 1
        thresh_higher = lifetimes[idx_thresh_2] + 1
    
    n = df.shape[0]
    n_short = np.sum(vals["lifetime"]<=thresh_lower)
    n_long = np.sum(vals["lifetime"]>=thresh_higher)
    acc_short = accs_cum_lower[idx_thresh]
    acc_long = accs_cum_higher[idx_thresh_2]

    if plot:
        plt.figure(figsize=(12, 4), dpi=200)
        plt.subplot(R, C, 1)
        outcome = df[outcome_def]
        plt.hist(df['lifetime'][outcome==1], label='aux+', alpha=1, color=cb, bins=25)
        plt.hist(df['lifetime'][outcome==0], label='aux-', alpha=0.7, color=cr, bins=25)
        plt.xlabel('lifetime')
        plt.ylabel('count')
        plt.legend()
    
        plt.subplot(R, C, 2)
        plt.plot(lifetimes, accs_cum_lower, color=cr)
    #     plt.axvline(thresh_lower)
        plt.axvspan(0, thresh_lower, alpha=0.2, color=cr)
        plt.ylabel('fraction of negative events')
        plt.xlabel(f'lifetime <= value\nshaded includes {n_short/n*100:0.0f}% of pts')


        plt.subplot(R, C, 3)
        plt.plot(lifetimes, accs_cum_higher, cb)
        plt.axvspan(thresh_higher, max(lifetimes), alpha=0.2, color=cb)
        plt.ylabel('fraction of positive events')
        plt.xlabel(f'lifetime >= value\nshaded includes {n_long/n*100:0.0f}% of pts')
        plt.tight_layout()
        plt.show()

    # only df with lifetimes in proper range
    df = df[(df['lifetime'] > thresh_lower) & (df['lifetime'] < thresh_higher)]
    metadata = {'num_short': n_short, 'num_long': n_long, 'acc_short': acc_short, 
                'acc_long': acc_long, 'thresh_short': thresh_lower, 'thresh_long': thresh_higher,
                'num_tracks_after_lifetime': df.shape[0], 'num_aux_pos_after_lifetime': df[outcome_def].sum(),
                'num_hotspots_after_lifetime': df['hotspots'].sum()
               }
    return df, metadata

def add_dict_features(df, sc_comps_file='processed/dictionaries/sc_12_alpha=1.pkl', 
                      nmf_comps_file='processed/dictionaries/nmf_12.pkl', 
                      use_processed=True):
    
    '''Add features from saved dictionary to df
    '''
    def sparse_code(X_mat, n_comps=12, alpha=1, out_dir='processed/dictionaries'):
        print('sparse coding...')
        d = DictionaryLearning(n_components=n_comps, alpha=alpha, random_state=42)
        d.fit(X_mat)
        pkl.dump(d, open(oj(out_dir, f'sc_{n_comps}_alpha={alpha}.pkl'), 'wb'))

    def nmf(X_mat, n_comps=12, out_dir='processed/dictionaries'):
        print('running nmf...')
        d = NMF(n_components=n_comps, random_state=42)
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
                   
    
def get_feature_names(df):
    '''Returns features (all of which are scalar)
    Removes metadata + time-series columns + outcomes
    '''
    ks = list(df.keys())
    feat_names = [
        k for k in ks
        if not k.startswith('y')
        and not k.startswith('Y')
        and not k.startswith('pixel')
#         and not k.startswith('pc_')
        and not k in ['catIdx', 'cell_num', 'pid', # metadata
                      'X', 'X_pvals', 'x_pos',
                      'X_peak_idx',
                      'hotspots', 'sig_idxs',
                      'X_smooth_spl', 'X_smooth_spl_dx', 'X_smooth_spl_d2x'] # curves not features
    ]
    return feat_names

def add_pcs(df):
    '''adds 10 pcs based on feature names
    '''
    feat_names = get_feature_names(df)
    X = df[feat_names]
    X = (X - X.mean()) / X.std()
    pca = decomposition.PCA(whiten=True)
    pca.fit(X[~df.cell_num.isin(cell_nums_test)])
    X_reduced = pca.transform(X)  
    for i in range(10):
        df['pc_' + str(i)] = X_reduced[:, i]
    return df