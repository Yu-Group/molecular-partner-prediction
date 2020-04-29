from matplotlib import pyplot as plt
import os
from os.path import join as oj
from skimage.external.tifffile import imread
from sklearn.linear_model import LinearRegression, RidgeCV
import numpy as np
from copy import deepcopy
import mat4py
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn' - caution: this turns off setting with copy warning
import pickle as pkl
from style import *
import math
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import DictionaryLearning, NMF
from sklearn import decomposition
import trend_filtering
import config




def get_data(dset='orig', use_processed=True, save_processed=True, processed_file='processed/df.pkl',
             metadata_file='processed/metadata.pkl', use_processed_dicts=True,
             outcome_def='y_consec_thresh', all_data=False, acc_thresh=0.95):
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
    # get things based onn dset
    DATA_DIR = config.DATA_DIRS[dset]
    SPLIT = config.SPLITS[dset]
    LABELS = config.LABELS[dset]
    
    processed_file = processed_file[:-4] + '_' + dset + '.pkl'
    metadata_file = metadata_file[:-4] + '_' + dset + '.pkl'
    
    
    if use_processed and os.path.exists(processed_file):
        return pd.read_pickle(processed_file)
    else:
        print('loading + preprocessing data...')
        metadata = {}
        print('\tloading tracks...')
        df = get_tracks(data_dir=DATA_DIR, split=SPLIT, all_data=all_data, dset=dset)  # note: different Xs can be different shapes
        df['pid'] = np.arange(df.shape[0])  # assign each track a unique id
        df['valid'] = True  # all tracks start as valid
        df['valid'][df.cell_num.isin(SPLIT['test'])] = False
        metadata['num_tracks'] = df.valid.sum()

        print('\tpreprocessing data...')
        df = remove_invalid_tracks(df) # use catIdx
        df = add_features(df)
        df = add_outcomes(df, LABELS=LABELS)

        metadata['num_tracks_valid'] = df.valid.sum()
        metadata['num_aux_pos_valid'] = df[df.valid][outcome_def].sum()
        metadata['num_hotspots_valid'] = df[df.valid]['hotspots'].sum()
        df['valid'][df.hotspots] = False
        df, meta_lifetime = process_tracks_by_lifetime(df, outcome_def=outcome_def,
                                                       plot=False, acc_thresh=acc_thresh)
        df['valid'][df.short] = False
        df['valid'][df.long] = False
        metadata.update(meta_lifetime)
        metadata['num_tracks_hard'] = df['valid'].sum()
        metadata['num_aux_pos_hard'] = int(df[df.valid == 1][outcome_def].sum())

        print('\tadding features...')
        # df = add_dict_features(df, use_processed=use_processed_dicts)
        # df = add_smoothed_tracks(df)
        df = add_pcs(df)
        df = add_trend_filtering(df)
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
    print(processed_tracks_file, data_dir)
    
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
            print(cell_num)
            fname = f'{data_dir}/{upper_dir}/{cell_dir}/TagRFP/Tracking/ProcessedTracks.mat'
            cla, aux = get_images(f'{data_dir}/{upper_dir}/{cell_dir}')
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
            pixel_up = np.array([[cla[int(t[i][j]), min(int(y_pos_seq[i][j] + 1), cla.shape[1] - 1), int(x_pos_seq[i][j])]
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


def add_features(df):
    '''Add a bunch of extra features to the df based on df.X, df.X_extended, df.Y, df.lifetime
    '''
    df = df[df.lifetime > 2]
    df['X_max'] = np.array([max(x) for x in df.X.values])
    df['X_max_extended'] = np.array([max(x) for x in df.X_extended.values])
    df['X_min'] = np.array([min(x) for x in df.X.values])
    df['X_mean'] = np.nan_to_num(np.array([np.nanmean(x) for x in df.X.values]))
    df['X_std'] = np.nan_to_num(np.array([np.std(x) for x in df.X.values]))
    df['Y_max'] = np.array([max(y) for y in df.Y.values])
    df['Y_mean'] = np.nan_to_num(np.array([np.nanmean(y) for y in df.Y.values]))
    df['Y_std'] = np.nan_to_num(np.array([np.std(y) for y in df.Y.values]))
    df['X_peak_idx'] = np.nan_to_num(np.array([np.argmax(x) for x in df.X]))
    df['Y_peak_idx'] = np.nan_to_num(np.array([np.argmax(y) for y in df.Y]))
    df['X_peak_time_frac'] = df['X_peak_idx'].values / df['lifetime'].values
    df['slope_end'] = df.apply(lambda row: (row['X_max'] - row['X'][-1]) / (row['lifetime'] - row['X_peak_idx']),
                               axis=1)
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
        x_late = x[idx_max:]
        idx_min = np.argmin(x_late)
        denom = idx_min
        if denom == 0:
            return 0
        return (val_max - np.min(x_late)) / denom

    def max_diff(x):
        return np.max(np.diff(x))

    def min_diff(x):
        return np.min(np.diff(x))

    df['rise'] = df.apply(lambda row: calc_rise(row['X']), axis=1)
    df['fall'] = df.apply(lambda row: calc_fall(row['X']), axis=1)
    df['rise_extended'] = df.apply(lambda row: calc_rise(row['X_extended']), axis=1)
    df['fall_extended'] = df.apply(lambda row: calc_fall(row['X_extended']), axis=1)
    df['fall_late_extended'] = df.apply(lambda row: row['fall_extended'] if row['X_peak_last_15'] else row['fall'],
                                        axis=1)
    df['fall_final'] = df.apply(lambda row: row['X'][-3] - row['X'][-1], axis=1)

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

    # imputed feats
    d = df[['X_max', 'X_mean', 'lifetime', 'rise', 'fall']]
    d = d[df['X_peak_time_frac'] <= 0.8]
    m = RidgeCV().fit(d[['X_max', 'X_mean', 'lifetime', 'rise']], d['fall'])
    fall_pred = m.predict(df[['X_max', 'X_mean', 'lifetime', 'rise']])
    fall_imp = df['fall']
    fall_imp[df['X_peak_time_frac'] > 0.8] = fall_pred[df['X_peak_time_frac'] > 0.8]
    df['fall_imp'] = fall_imp

    return df


def add_rule_based_label(df):
    df['Y_peak_time_frac'] = df['Y_peak_idx'].values / df['lifetime'].values
    df['y_z_score'] = (df['Y_max'].values - df['Y_mean'].values) / df['Y_std'].values
    X_max_around_Y_peak = []
    X_max_after_Y_peak = []
    for i in range(len(df)):
        pt = df['Y_peak_idx'].values[i]
        lt = df['lifetime'].values[i]
        left_bf = np.int(0.2 * lt) + 1  # look at a window with length = 30%*lifetime
        right_bf = np.int(0.1 * lt) + 1
        arr_around = df['X'].iloc[i][max(0, pt - left_bf): min(pt + right_bf, lt)]
        arr_after = df['X'].iloc[i][min(pt + right_bf, lt - 1):]
        X_max_around_Y_peak.append(max(arr_around))
        X_max_after_Y_peak.append(max(arr_after))
    df['X_max_around_Y_peak'] = X_max_around_Y_peak
    df['X_max_after_Y_peak'] = X_max_after_Y_peak
    df['X_max_diff'] = df['X_max_around_Y_peak'] - df['X_max_after_Y_peak']

    def rule_based_model(track):

        # three rules:
        #  if aux peaks too early -- negative
        #  elif:
        #     if y_consec_sig or y_conservative_thresh or (cla drops around aux peak, and aux max is greater than 
        #     mean + 2.6*std), then positive
        #  else: negative

        if track['Y_peak_time_frac'] < 0.2:
            return 0
        if track['y_consec_sig'] or track['y_conservative_thresh']:
            return 1
        # if track['X_max_diff'] > 260 and track['y_z_score'] > 2.6:
        #    return 1
        if track['X_max_diff'] > 260 and track['Y_max'] > 560:
            return 1
        return 0

    df['y_rule_based'] = np.array([rule_based_model(df.iloc[i]) for i in range(len(df))])
    return df


def add_outcomes(df, LABELS=None, thresh=3.25, p_thresh=0.05, aux_peak=642.375, aux_thresh=973):
    '''Add binary outcome of whether spike happened and info on whether events were questionable
    '''
    df['y_score'] = df['Y_max'].values - (df['Y_mean'].values + thresh * df['Y_std'].values)
    df['y_thresh'] = (df['y_score'].values > 0).astype(np.int)  # Y_max was big
    df['y'] = df['Y_max'] > aux_peak

    # outcomes based on significant p-values
    num_sigs = [np.array(df['Y_pvals'].iloc[i]) < p_thresh for i in range(df.shape[0])]
    df['y_num_sig'] = np.array([num_sigs[i].sum() for i in range(df.shape[0])]).astype(np.int)
    df['y_single_sig'] = np.array([num_sigs[i].sum() > 0 for i in range(df.shape[0])]).astype(np.int)
    df['y_double_sig'] = np.array([num_sigs[i].sum() > 1 for i in range(df.shape[0])]).astype(np.int)
    df['y_conservative_thresh'] = (df['Y_max'].values > aux_thresh).astype(np.int)
    y_consec_sig = []
    y_sig_min_diff = []
    for i in range(df.shape[0]):
        idxs_sig = np.where(num_sigs[i] == 1)[0]  # indices of significance
        if len(idxs_sig) > 1:
            y_sig_min_diff.append(np.min(np.diff(idxs_sig)))
        else:
            y_sig_min_diff.append(np.nan)
        # find whether there were consecutive sig. indices
        if len(idxs_sig) > 1 and np.min(np.diff(idxs_sig)) == 1:
            y_consec_sig.append(1)
        else:
            y_consec_sig.append(0)
    df['y_consec_sig'] = y_consec_sig
    df['y_sig_min_diff'] = y_sig_min_diff
    df['y_consec_thresh'] = np.logical_or(df['y_consec_sig'], df['y_conservative_thresh'])

    def add_hotspots(df, num_sigs, outcome_def='consec_sig'):
        '''Identify hotspots as any track which over its time course has multiple events
        events must meet the event definition, then for a time not meet it, then meet it again
        Example: two consecutive significant p-values, then non-significant p-value, then 2 more consecutive p-values
        '''

        if outcome_def == 'consec_sig':
            hotspots = np.zeros(df.shape[0]).astype(np.int)
            for i in range(df.shape[0]):
                idxs_sig = np.where(num_sigs[i] == 1)[0]  # indices of significance
                if idxs_sig.size < 5:
                    hotspots[i] = 0
                else:
                    diffs = np.diff(idxs_sig)
                    consecs = np.where(diffs == 1)[0]  # diffs==1 means there were consecutive sigs
                    consec_diffs = np.diff(consecs)
                    if consec_diffs.shape[0] > 0 and np.max(
                            consec_diffs) > 2:  # there were greated than 2 non-consec sigs between the consec sigs
                        hotspots[i] = 1
                    else:
                        hotspots[i] = 0
        df['sig_idxs'] = num_sigs
        df['hotspots'] = hotspots == 1

        return df

    df = add_hotspots(df, num_sigs)

    df['y_consec_thresh'][df.pid.isin(LABELS['pos'])] = 1  # add manual pos labels
    df['y_consec_thresh'][df.pid.isin(LABELS['neg'])] = 0  # add manual neg labels
    df['hotspots'][df.pid.isin(LABELS['hotspots'])] = True  # add manual hotspot labels

    df = add_rule_based_label(df)

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


def process_tracks_by_lifetime(df: pd.DataFrame, outcome_def: str, plot=False, acc_thresh=0.95):
    '''Calculate accuracy you can get by just predicting max class 
    as a func of lifetime and return points within proper lifetime (only looks at training cells)
    '''
    vals = df[df.valid == 1][['lifetime', outcome_def]]

    R, C = 1, 3
    lifetimes = np.unique(vals['lifetime'])

    # cumulative accuracy for different thresholds
    accs_cum_lower = np.array([1 - np.mean(vals[outcome_def][vals['lifetime'] <= l]) for l in lifetimes])
    idx_thresh = np.nonzero(accs_cum_lower >= acc_thresh)[0][-1]  # last nonzero index
    thresh_lower = lifetimes[idx_thresh]

    accs_cum_higher = np.array([np.mean(vals[outcome_def][vals['lifetime'] >= l]) for l in lifetimes]).flatten()
    try:
        idx_thresh_2 = np.nonzero(accs_cum_higher >= acc_thresh)[0][0]
        thresh_higher = lifetimes[idx_thresh_2]
    except:
        idx_thresh_2 = lifetimes.size - 1
        thresh_higher = lifetimes[idx_thresh_2] + 1

    n = vals.shape[0]
    n_short = np.sum(vals["lifetime"] <= thresh_lower)
    n_long = np.sum(vals["lifetime"] >= thresh_higher)
    acc_short = accs_cum_lower[idx_thresh]
    acc_long = accs_cum_higher[idx_thresh_2]

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

    # only df with lifetimes in proper range
    df['short'] = df['lifetime'] <= thresh_lower
    df['long'] = df['lifetime'] >= thresh_higher
    metadata = {'num_short': n_short, 'num_long': n_long, 'acc_short': acc_short,
                'acc_long': acc_long, 'thresh_short': thresh_lower, 'thresh_long': thresh_higher}
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


def add_smoothed_splines(df,
                         method='spline',
                         s_spl=0.004):
    X_smooth_spl = []
    X_smooth_spl_dx = []
    X_smooth_spl_d2x = []

    def num_local_maxima(x):
        return (len([i for i in range(1, len(x) - 1) if x[i] > x[i - 1] and x[i] > x[i + 1]]))

    for x in df['X']:
        spl = UnivariateSpline(x=range(len(x)),
                               y=x,
                               w=[1.0 / len(x)] * len(x),
                               s=np.var(x) * s_spl)
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

    # linear fits
    x = np.arange(5).reshape(-1, 1)
    df['end_linear_fit'] = [LinearRegression().fit(x, end).coef_[0] for end in df['X_ends']]
    df['start_linear_fit'] = [LinearRegression().fit(x, start).coef_[0] for start in df['X_starts']]
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
           and not k in ['catIdx', 'cell_num', 'pid', 'valid',  # metadata
                         'X', 'X_pvals', 'x_pos', 'X_starts', 'X_ends', 'X_extended',
                         'X_peak_idx', 'short', 'long',
                         'hotspots', 'sig_idxs',
                         'X_max_around_Y_peak',
                         'X_max_after_Y_peak',
                         'X_max_diff',
                         't', 'x_pos_seq', 'y_pos_seq',
                         'X_smooth_spl', 'X_smooth_spl_dx', 'X_smooth_spl_d2x']  # curves not features
    ]
    return feat_names


def add_pcs(df):
    '''adds 10 pcs based on feature names
    '''
    feat_names = get_feature_names(df)
    X = df[feat_names]
    X = (X - X.mean()) / X.std()
    pca = decomposition.PCA(whiten=True)
    pca.fit(X[df.valid])
    X_reduced = pca.transform(X)
    for i in range(10):
        df['pc_' + str(i)] = X_reduced[:, i]
    return df


def add_trend_filtering(df):
    df_tf = deepcopy(df)
    for i in range(len(df)):
        df_tf['X'].iloc[i] = trend_filtering.trend_filtering(y=df['X'].iloc[i], vlambda=len(df['X'].iloc[i]) * 5,
                                                             order=1)
    df_tf = add_features(df_tf)
    feat_names = get_feature_names(df_tf)
    feat_names = [x for x in feat_names
                  if not x.startswith('sc_')
                  and not x.startswith('nmf_')
                  and not x in ['center_max', 'left_max', 'right_max', 'up_max', 'down_max',
                                'X_max_around_Y_peak', 'X_max_after_Y_peak', 'X_max_diff_after_Y_peak',
                                'X_tf']
                  and not x.startswith('pc_')
                  #               and not 'local' in x
                  #               and not 'X_peak' in x
                  #               and not 'slope' in x
                  #               and not x in ['fall_final', 'fall_slope', 'fall_imp', 'fall']
                  ]
    for feat in feat_names:
        df[feat + '_tf_smooth'] = df_tf[feat]
    return df
