#!/usr/bin/python3
import os
from os.path import join as oj
import sys
sys.path.append('../src')
import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
plt.style.use('dark_background')
import data
from tqdm import tqdm
import train_reg
import config
import pandas as pd
import features
from scipy.stats import skew, pearsonr
import outcomes
from sklearn.model_selection import KFold
from collections import defaultdict

datasets = ['clath_aux+gak_a7d2', 'clath_aux+gak_a7d2_new', 'clath_aux+gak_new', 'clath_gak']
orig_dset = 'clath_aux+gak_a7d2'
outcome_def = 'Y_sig_mean'

def load_all_datasets():
    
    all_data = {}
    for d in datasets:
        all_data[d] = data.get_data(dset=d)
        all_data[d] = train_reg.add_sig_mean(all_data[d], resp_tracks=['Y'])
        all_data[d] = features.add_dasc_features(all_data[d], bins=100, by_cell=True)
    
    return all_data

def train_and_save_models(df_train, feat_names, outcome_def):
    
    for features in ['dasc', 'basic', 'combined']:
        if features == 'dasc':
            use_feature = ['X_d1', 'X_d2', 'X_d3']
        elif features == 'basic':
            use_feature = feat_names
        else:
            use_feature = feat_names + ['X_d1', 'X_d2', 'X_d3']
        #df_train = df[df.cell_num.isin(config.DSETS[dset]['train'])] 
        out_dir = f'results/regression/{orig_dset}_Y_{features}_{outcome_def}'
        os.makedirs(out_dir, exist_ok=True)
        for model_type in tqdm(['linear', 'gb', 'rf', 'ridge']):
            out_name = f'{model_type}'

            train_reg.train_reg(df_train, feat_names=use_feature, model_type=model_type, 
                             outcome_def=outcome_def,
                             out_name=f'{out_dir}/{out_name}.pkl') 
            
def test_and_save_results(model_name='rf'):
    
    res = defaultdict(list)
    for d in all_data:
        #m = pd.read_pickle(f'{out_dir}/rf.pkl')
        if d == 'clath_aux_dynamin':
            continue
        for feat in ['dasc', 'basic', 'combined']:
            out_dir = f'results/regression/{orig_dset}_Y_{feat}_{outcome_def}'
            m = pd.read_pickle(f'{out_dir}/{model_name}.pkl')
            if feat == 'dasc':
                use_feature = ['X_d1', 'X_d2', 'X_d3']
            elif feat == 'basic':
                use_feature = feat_names
            else:
                use_feature = feat_names + ['X_d1', 'X_d2', 'X_d3']
            df_test = all_data[d]
            if d == 'clath_aux+gak_a7d2':
                df_test =  df_test[df_test.cell_num.isin(config.DSETS[d]['test'])] 
            #df_test = features.add_dasc_features(df_test, bins=100, by_cell=True)
            df_test = df_test[use_feature + [outcome_def]]#[df_test['valid'] == 1]
            df_test = df_test.fillna(df_test.mean())
            if len(df_test) > 0:
                test_preds = train_reg.test_reg(df_test, m['model'], feat_names=use_feature, outcome_def=outcome_def)
            #print(f"R^2 with features: {test_preds['r2']}")
            #print(f"corr with features: {test_preds['pearsonr']}")
            for metric in ['r2', 'pearsonr', 'kendalltau']:
                if metric == 'r2':
                    res[f'{feat}_{metric}'].append(test_preds[metric])
                else:
                    res[f'{feat}_{metric}'].append(test_preds[metric][0])
                    
    res_df = pd.DataFrame(res, index=datasets)
    res_df.to_pickle('results/regression/regression_results.pkl')
            
if __name__ == '__main__':
    
    print("loading data...")
    all_data = load_all_datasets()
    print("training regression models...")
    df = all_data[orig_dset]
    df_train = df[df.cell_num.isin(config.DSETS[orig_dset]['train'])] 
    df_train = df_train[df_train['valid'] == 1]
    feat_names = data.get_feature_names(df_train)
    feat_names = [x for x in feat_names 
                      if not x.startswith('sc_') 
                      and not x.startswith('nmf_')
                      and not x in ['center_max', 'left_max', 'right_max', 'up_max', 'down_max',
                                   'X_max_around_Y_peak', 'X_max_after_Y_peak', 'X_quantiles',
                                   'X_d1', 'X_d2', 'X_d3', 'slope_end'
                                   ]
                      and not x.startswith('pc_')
                      and not 'log' in x
                      and not 'binary' in x
        #               and not 'slope' in x
                     ]
    train_and_save_models(df_train, feat_names, outcome_def)
    print("testing regression models...")
    test_and_save_results()
    print("Testing succesfully completed")
    