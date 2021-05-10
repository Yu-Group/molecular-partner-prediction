import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn' - caution: this turns off setting with copy warning
from viz import *


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
        if len(arr_after) > 0:
            X_max_after_Y_peak.append(max(arr_after))
        else:
            X_max_after_Y_peak.append(max(arr_around))
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

    if LABELS is not None:
        df['y_consec_thresh'][df.pid.isin(LABELS['pos'])] = 1  # add manual pos labels
        df['y_consec_thresh'][df.pid.isin(LABELS['neg'])] = 0  # add manual neg labels
        df['hotspots'][df.pid.isin(LABELS['hotspots'])] = True  # add manual hotspot labels

    df = add_rule_based_label(df)

    return df

def add_sig_mean(df, resp_tracks=['Y']):
    """add response of regression problem: mean auxilin strength among significant observations
    """
    for track in resp_tracks:
        sig_mean = []
        for i in range(len(df)):
            r = df.iloc[i]
            sigs = np.array(r[f'{track}_pvals']) < 0.05
            if sum(sigs)>0:
                sig_mean.append(np.mean(np.array(r[track])[sigs]))
            else:
                sig_mean.append(0)
        df[f'{track}_sig_mean'] = sig_mean
        df[f'{track}_sig_mean_normalized'] = sig_mean
        for cell in set(df['cell_num']):
            cell_idx = np.where(df['cell_num'].values == cell)[0]
            y = df[f'{track}_sig_mean'].values[cell_idx]
            df[f'{track}_sig_mean_normalized'].values[cell_idx] = (y - np.mean(y))/np.std(y)
    return df

def add_aux_dyn_outcome(df, p_thresh=0.05, clath_thresh=1500, dyn_thresh=1500):
    """add response of regression problem: mean auxilin strength among significant observations
    """
    # outcomes based on significant p-values
    
    df['clath_conservative_thresh'] = (df['X_max'].values > clath_thresh).astype(np.int)
    df['successful'] = np.logical_and(df['y_consec_thresh'], df['clath_conservative_thresh'])

    # look for dynamin peak
    if 'Z' in df.keys():
        num_sigs = [np.array(df['Z_pvals'].iloc[i]) < p_thresh for i in range(df.shape[0])]
        z_consec_sig = []
        for i in range(df.shape[0]):
            idxs_sig = np.where(num_sigs[i] == 1)[0]  # indices of significance
                
            # find whether there were consecutive sig. indices
            if len(idxs_sig) > 1 and np.min(np.diff(idxs_sig)) == 1:
                z_consec_sig.append(1)
            else:
                z_consec_sig.append(0)
        df['z_consec_sig'] = z_consec_sig
        df['Z_max'] = [np.max(df.iloc[i]['Z']) for i in range(df.shape[0])]
        df['z_thresh'] = df['Z_max'] > dyn_thresh
        df['z_consec_thresh'] = np.logical_and(df['z_consec_sig'], df['z_thresh'])
        
        df['successful_dynamin'] = np.logical_or(df['y_consec_thresh'],
                                         np.logical_and(df['clath_conservative_thresh'],
                                                        df['z_consec_thresh']))
    return df