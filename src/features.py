import os
from copy import deepcopy
from os.path import join as oj

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV

pd.options.mode.chained_assignment = None  # default='warn' - caution: this turns off setting with copy warning
import pickle as pkl
from viz import *
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import DictionaryLearning, NMF
from sklearn import decomposition
import trend_filtering
import data


def add_pcs(df):
    '''adds 10 pcs based on feature names
    '''
    feat_names = data.get_feature_names(df)
    X = df[feat_names]
    X = (X - X.mean()) / X.std()
    pca = decomposition.PCA(whiten=True)
    pca.fit(X[df.valid])
    X_reduced = pca.transform(X)
    for i in range(10):
        df['pc_' + str(i)] = X_reduced[:, i]
    return df


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


def add_trend_filtering(df):
    df_tf = deepcopy(df)
    for i in range(len(df)):
        df_tf['X'].iloc[i] = trend_filtering.trend_filtering(y=df['X'].iloc[i], vlambda=len(df['X'].iloc[i]) * 5,
                                                             order=1)
    df_tf = add_features(df_tf)
    feat_names = data.get_feature_names(df_tf)
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


def add_basic_features(df):
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
#     m = RidgeCV().fit(d[['X_max', 'X_mean', 'lifetime', 'rise']], d['fall'])
#     fall_pred = m.predict(df[['X_max', 'X_mean', 'lifetime', 'rise']])
#     fall_imp = df['fall']
#     fall_imp[df['X_peak_time_frac'] > 0.8] = fall_pred[df['X_peak_time_frac'] > 0.8]
#     df['fall_imp'] = fall_imp

    return df


def add_binary_features(df, outcome_def):
    '''binarize features at the difference between the mean of each class
    '''
    feat_names = data.get_feature_names(df)
    threshes = (df[df[outcome_def] == 1].mean() + df[df[outcome_def] == 0].mean()) / 2
    for i, k in enumerate(feat_names):
        thresh = threshes.loc[k]
        df[k + '_binary'] = df[k] >= thresh
    return df
