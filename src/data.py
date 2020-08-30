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
import features
import outcomes


def get_data(dset='clath_aux+gak_a7d2', use_processed=True, save_processed=True, processed_file='processed/df.pkl',
             metadata_file='processed/metadata.pkl', use_processed_dicts=True,
             outcome_def='y_consec_thresh', all_data=False, acc_thresh=0.95,
             previous_meta_file=None):
    '''
    Params
    ------
    use_processed: bool, optional
        determines whether to load df from cached pkl
    save_processed: bool, optional
        if not using processed, determines whether to save the df
    use_processed_dicts: bool, optional
        if False, recalculate the dictionary features
    previous_meta_file: str, optional
        filename for metadata.pkl file saved by previous preprocessing
        the thresholds for lifetime are taken from this file
    '''
    # get things based onn dset
    DSET = config.DSETS[dset]
    LABELS = config.LABELS[dset]

    processed_file = processed_file[:-4] + '_' + dset + '.pkl'
    metadata_file = metadata_file[:-4] + '_' + dset + '.pkl'

    if use_processed and os.path.exists(processed_file):
        return pd.read_pickle(processed_file)
    else:
        print('loading + preprocessing data...')
        metadata = {}
        print('\tloading tracks...')
        df = get_tracks(data_dir=DSET['data_dir'], split=DSET, all_data=all_data,
                        dset=dset)  # note: different Xs can be different shapes
        df['pid'] = np.arange(df.shape[0])  # assign each track a unique id
        df['valid'] = True  # all tracks start as valid
        df['valid'][df.cell_num.isin(DSET['test'])] = False
        metadata['num_tracks'] = df.valid.sum()

        print('\tpreprocessing data...')
        df = remove_invalid_tracks(df)  # use catIdx
        df = features.add_basic_features(df)
        df = outcomes.add_outcomes(df, LABELS=LABELS)

        metadata['num_tracks_valid'] = df.valid.sum()
        metadata['num_aux_pos_valid'] = df[df.valid][outcome_def].sum()
        metadata['num_hotspots_valid'] = df[df.valid]['hotspots'].sum()
        df['valid'][df.hotspots] = False
        df, meta_lifetime = process_tracks_by_lifetime(df, outcome_def=outcome_def,
                                                       plot=False, acc_thresh=acc_thresh,
                                                       previous_meta_file=previous_meta_file)
        df['valid'][df.short] = False
        df['valid'][df.long] = False
        metadata.update(meta_lifetime)
        metadata['num_tracks_hard'] = df['valid'].sum()
        metadata['num_aux_pos_hard'] = int(df[df.valid == 1][outcome_def].sum())

        print('\tadding features...')
        # df = features.add_dict_features(df, use_processed=use_processed_dicts)
        # df = features.add_smoothed_tracks(df)
        df = features.add_pcs(df)
        # df = features.add_trend_filtering(df) 
        df = features.add_binary_features(df, outcome_def=outcome_def)
        if save_processed:
            pkl.dump(metadata, open(metadata_file, 'wb'))
            df.to_pickle(processed_file)
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


def get_tracks(data_dir, split=None, all_data=False, processed_tracks_file='processed/tracks.pkl', dset='orig'):
    '''Read out tracks from folders within data_dir, assuming tracking has been done
    '''
    processed_tracks_file = processed_tracks_file[:-4] + '_' + dset + '.pkl'
    print('\t', processed_tracks_file, data_dir)

    if os.path.exists(processed_tracks_file):
        return pd.read_pickle(processed_tracks_file)
    dfs = []

    if split is not None:
        flatten = lambda l: [item for sublist in l for item in sublist]
        split = flatten(split.values())

    # 2 directories of naming
    for upper_dir in sorted(os.listdir(data_dir)):
        if '.' in upper_dir or 'Icon' in upper_dir:
            continue
        for cell_dir in sorted(os.listdir(oj(data_dir, upper_dir))):
            if not 'Cell' in cell_dir:
                continue
            cell_num = oj(upper_dir, cell_dir.replace('Cell', '').replace('_1s', ''))
            if split is not None:
                if not cell_num in split:
                    continue
            full_dir = f'{data_dir}/{upper_dir}/{cell_dir}'
            fname = full_dir + '/TagRFP/Tracking/ProcessedTracks.mat'
            print(cell_num)
            cla, aux = get_images(full_dir)
            # fname_image = oj(data_dir, upper_dir, cell_dir)
            mat = mat4py.loadmat(fname)
            tracks = mat['tracks']
            n = len(tracks['t'])
            totalDisplacement = []
            msd = []  # mean squared displacement
            for i in range(n):
                try:
                    totalDisplacement.append(tracks['MotionAnalysis'][i]['totalDisplacement'])
                except:
                    totalDisplacement.append(0)
                try:
                    msd.append(np.nanmax(tracks['MotionAnalysis'][i]['MSD']))
                except:
                    msd.append(0)

            CLATH = 0
            AUX = 1

            X = np.array([tracks['A'][i][CLATH] for i in range(n)])
            Y = np.array([tracks['A'][i][AUX] for i in range(n)])
            t = np.array([tracks['t'][i] for i in range(n)])
            x_pos_seq = np.array(
                [tracks['x'][i][CLATH] for i in range(n)])  # x-position for clathrin (auxilin is very similar)
            y_pos_seq = np.array(
                [tracks['y'][i][CLATH] for i in range(n)])  # y-position for clathrin (auxilin is very similar)
            X_pvals = np.array([tracks['pval_Ar'][i][CLATH] for i in range(n)])
            Y_pvals = np.array([tracks['pval_Ar'][i][AUX] for i in range(n)])

            # buffers
            X_starts = []
            Y_starts = []
            for d in tracks['startBuffer']:
                if len(d) == 0:
                    X_starts.append([])
                    Y_starts.append([])
                else:
                    X_starts.append(d['A'][CLATH])
                    Y_starts.append(d['A'][AUX])
            X_ends = []
            Y_ends = []
            for d in tracks['endBuffer']:
                if len(d) == 0:
                    X_ends.append([])
                    Y_ends.append([])
                else:
                    X_ends.append(d['A'][CLATH])
                    Y_ends.append(d['A'][AUX])
            X_extended = [X_starts[i] + X[i] + X_ends[i] for i in range(n)]

            # image feats
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

            data = {
                'X': X,
                'X_extended': X_extended,
                'Y': Y,
                'X_starts': X_starts,
                'Y_starts': Y_starts,
                'X_ends': X_ends,
                'Y_ends': Y_ends,
                'X_pvals': X_pvals,
                'Y_pvals': Y_pvals,
                'catIdx': tracks['catIdx'],
                'mean_total_displacement': [totalDisplacement[i] / tracks['lifetime_s'][i] for i in range(n)],
                'mean_square_displacement': msd,
                'lifetime': tracks['lifetime_s'],
                'lifetime_extended': [len(x) for x in X_extended],
                'x_pos': [sum(x) / len(x) for x in x_pos_seq],  # mean position in the image
                'y_pos': [sum(y) / len(y) for y in y_pos_seq],
                'cell_num': [cell_num] * n,
                't': [t[i][0] for i in range(n)],
                'x_pos_seq': x_pos_seq,
                'y_pos_seq': y_pos_seq,
            }
            if all_data:
                data['t'] = [t[i][0] for i in range(n)]
                data['pixel'] = pixel
                data['pixel_left'] = pixel_left
                data['pixel_right'] = pixel_right
                data['pixel_up'] = pixel_up
                data['pixel_down'] = pixel_down
                data['center_max'] = [max(pixel[i]) for i in range(n)],
                data['left_max'] = [max(pixel_left[i]) for i in range(n)],
                data['right_max'] = [max(pixel_right[i]) for i in range(n)],
                data['up_max'] = [max(pixel_up[i]) for i in range(n)],
                data['down_max'] = [max(pixel_down[i]) for i in range(n)],
            df = pd.DataFrame.from_dict(data)
            dfs.append(deepcopy(df))
    df = pd.concat(dfs)
    os.makedirs(os.path.dirname(processed_tracks_file), exist_ok=True)
    df.to_pickle(processed_tracks_file)
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
    df['lifetime_ref'] = [len(x) for x in df['X']]
    no_nan = df['lifetime'] == df['lifetime_ref']
    df = df[no_nan]
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


def process_tracks_by_lifetime(df: pd.DataFrame, outcome_def: str,
                               plot=False, acc_thresh=0.95, previous_meta_file=None):
    '''Calculate accuracy you can get by just predicting max class 
    as a func of lifetime and return points within proper lifetime (only looks at training cells)
    '''
    vals = df[df.valid == 1][['lifetime', outcome_def]]

    R, C = 1, 3
    lifetimes = np.unique(vals['lifetime'])

    # cumulative accuracy for different thresholds
    accs_cum_lower = np.array([1 - np.mean(vals[outcome_def][vals['lifetime'] <= l]) for l in lifetimes])
    accs_cum_higher = np.array([np.mean(vals[outcome_def][vals['lifetime'] >= l]) for l in lifetimes]).flatten()

    if previous_meta_file is None:
        try:
            idx_thresh = np.nonzero(accs_cum_lower >= acc_thresh)[0][-1]  # last nonzero index
            thresh_lower = lifetimes[idx_thresh]
        except:
            idx_thresh = 0
            thresh_lower = lifetimes[idx_thresh] - 1
        try:
            idx_thresh_2 = np.nonzero(accs_cum_higher >= acc_thresh)[0][0]
            thresh_higher = lifetimes[idx_thresh_2]
        except:
            idx_thresh_2 = lifetimes.size - 1
            thresh_higher = lifetimes[idx_thresh_2] + 1
    else:
        previous_meta = pkl.load(open(previous_meta_file, 'rb'))
        thresh_lower = previous_meta['thresh_short']
        thresh_higher = previous_meta['thresh_long']

    # only df with lifetimes in proper range
    df['short'] = df['lifetime'] <= thresh_lower
    df['long'] = df['lifetime'] >= thresh_higher
    n = vals.shape[0]
    n_short = np.sum(df['short'])
    n_long = np.sum(df['long'])
    acc_short = 1 - np.mean(vals[outcome_def][vals['lifetime'] <= thresh_lower])
    acc_long = np.mean(vals[outcome_def][vals['lifetime'] >= thresh_higher])

    metadata = {'num_short': n_short, 'num_long': n_long, 'acc_short': acc_short,
                'acc_long': acc_long, 'thresh_short': thresh_lower, 'thresh_long': thresh_higher}

    if plot:
        plt.figure(figsize=(12, 4), dpi=200)
        plt.subplot(R, C, 1)
        outcome = df[outcome_def]
        plt.hist(df['lifetime'][outcome == 1], label='aux+', alpha=1, color=cb, bins=25)
        plt.hist(df['lifetime'][outcome == 0], label='aux-', alpha=0.7, color=cr, bins=25)
        plt.xlabel('lifetime')
        plt.ylabel('count')
        plt.legend()

        plt.subplot(R, C, 2)
        plt.plot(lifetimes, accs_cum_lower, color=cr)
        #     plt.axvline(thresh_lower)
        plt.axvspan(0, thresh_lower, alpha=0.2, color=cr)
        plt.ylabel('fraction of negative events')
        plt.xlabel(f'lifetime <= value\nshaded includes {n_short / n * 100:0.0f}% of pts')

        plt.subplot(R, C, 3)
        plt.plot(lifetimes, accs_cum_higher, cb)
        plt.axvspan(thresh_higher, max(lifetimes), alpha=0.2, color=cb)
        plt.ylabel('fraction of positive events')
        plt.xlabel(f'lifetime >= value\nshaded includes {n_long / n * 100:0.0f}% of pts')
        plt.tight_layout()

    return df, metadata


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
           and not k in ['catIdx', 'cell_num', 'pid', 'valid',  # metadata
                         'X', 'X_pvals', 'x_pos', 'X_starts', 'X_ends', 'X_extended',  # curves
                         'short', 'long', 'hotspots', 'sig_idxs',  # should be weeded out
                         'X_max_around_Y_peak', 'X_max_after_Y_peak',  # redudant with X_max / fall
                         'X_max_diff', 'X_peak_idx',  # unlikely to be useful
                         't', 'x_pos_seq', 'y_pos_seq',  # curves
                         'X_smooth_spl', 'X_smooth_spl_dx', 'X_smooth_spl_d2x'  # curves
                         ]
    ]
    return feat_names


def select_final_feats(feat_names, binarize=False):
    feat_names = [x for x in feat_names
                  if not x.startswith('sc_')
                  and not x.startswith('nmf_')
                  and not x in ['center_max', 'left_max', 'right_max', 'up_max', 'down_max',
                                'X_max_around_Y_peak', 'X_max_after_Y_peak', 'X_max_diff_after_Y_peak']
                  and not x.startswith('pc_')
                  and not 'extended' in x
                  #               and not 'X_peak' in x
                  #               and not 'slope' in x
                  #               and not x in ['fall_final', 'fall_slope', 'fall_imp', 'fall']
                  ]
    feat_names = [x for x in feat_names if not '_tf_smooth' in x]
    feat_names = [x for x in feat_names if not 'local' in x]
    feat_names = [x for x in feat_names if not 'last' in x]
    # feat_names = [x for x in feat_names if '_tf_smooth' in x]

    if binarize:
        feat_names = [x for x in feat_names if 'binary' in x]
    else:
        feat_names = [x for x in feat_names if not 'binary' in x]
    return feat_names


if __name__ == '__main__':
    # process original data (and save out lifetime thresholds)
    dset_orig = 'clath_aux+gak_a7d2'
    df = get_data(dset=dset_orig)  # save out orig
    outcome_def = 'y_consec_sig'
    for dset in config.DSETS.keys():
        # process new data (using lifetime thresholds from original data)
        df = get_data(dset=dset,
                      previous_meta_file=f'processed/metadata_{dset_orig}.pkl')
        print(dset, 'num cells', len(df['cell_num'].unique()), 'num tracks', df.shape[0], 'num aux+',
              df[outcome_def].sum(), 'ratio', (df[outcome_def].sum() / df.shape[0]).round(3),
              'valid', df.valid.sum(), 'valid aux+', df[df.valid][outcome_def].sum(), 'ratio',
              (df[df.valid][outcome_def].sum() / df.valid.sum()).round(3))
