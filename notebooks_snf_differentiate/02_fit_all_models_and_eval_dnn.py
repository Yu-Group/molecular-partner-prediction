'''This script re-fits all non-LSTM models and evaluates them for each cell / dataset.
It also takes a pre-trained LSTM and evaluates it on each cell / dataset.
'''

import os
from os.path import join as oj
import sys

from sklearn.tree import DecisionTreeRegressor
sys.path.append('../src')
import numpy as np
import torch
import scipy
from matplotlib import pyplot as plt
from sklearn import metrics
import data
from config import *
from tqdm import tqdm
import pickle as pkl
import train_reg
from copy import deepcopy
import config
import models
import pandas as pd
import features
import outcomes
import neural_networks
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.nn import functional as F
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.svm import SVR
from collections import defaultdict

scorers = {
    'balanced_accuracy': metrics.balanced_accuracy_score,
    'accuracy': metrics.accuracy_score,
    'roc_auc': metrics.roc_auc_score,
    'r2': metrics.r2_score,
    'corr': scipy.stats.pearsonr,
    'recall': metrics.recall_score,
    'f1': metrics.f1_score
}

def get_all_scores(y, preds, y_reg, df):
    
    for metric in scorers:
        if 'accuracy' in metric or 'recall' in metric or 'f1' in metric:
            acc = scorers[metric](y, (preds > 0.5).astype(int))                   
            # print('curr', type(dataset_level_res[f'{k}_{metric}']), dataset_level_res[f'{k}_{metric}'])
            # print('new', type(acc), acc)

            dataset_level_res[f'{k}_{metric}'].append(acc)
        elif metric == 'roc_auc':
            dataset_level_res[f'{k}_{metric}'].append(scorers[metric](y, preds))
        elif metric == 'r2':
            dataset_level_res[f'{k}_{metric}'].append(scorers[metric](y_reg, preds))
        else:
            dataset_level_res[f'{k}_{metric}'].append(scorers[metric](y_reg, preds)[0])
                            
    # for cell in set(df['cell_num']):
    #     cell_idx = np.where(df['cell_num'].values == cell)[0]
    #     y_cell = y[cell_idx]
    #     y_reg_cell = y_reg[cell_idx]
    #     preds_cell = preds[cell_idx]
    #     for metric in scorers:
    #         if 'accuracy' in metric or 'recall' in metric or 'f1' in metric:
    #             acc = scorers[metric](y_cell, np.logical_and((preds_cell > 0), df['X_max_orig'].values[cell_idx] > 1500).astype(int))              
    #             cell_level_res[f'{cell}_{metric}'].append(acc)
    #         elif metric == 'roc_auc':
    #             cell_level_res[f'{cell}_{metric}'].append(scorers[metric](y_cell, preds_cell))
    #         elif metric == 'r2':
    #             cell_level_res[f'{cell}_{metric}'].append(scorers[metric](y_reg_cell, preds_cell))
    #         else:
    #             cell_level_res[f'{cell}_{metric}'].append(scorers[metric](y_reg_cell, preds_cell)[0])                       

if __name__ == '__main__':
    k = 'test' 
    df_train, df_test, feat_names = data.get_snf_mt_vs_wt()
    epoch = 100
    # feat_name = feat_names[0]
    feat_name = 'X_same_length'
    # feat_names = ['X_same_length_normalized'] # include buffer X_same_length_normalized
    # feat_names = ['X_same_length_normalized', 'lifetime', 'lifetime_s', 'mean_total_displacement', 'mean_square_displacement', 'X_max', 'X_min', 'X_mean', 'X_std', 'X_peak_time_frac', 'rise', 'fall', 'rise_slope', 'fall_slope', 'max_diff', 'min_diff', 'X_d1', 'X_d2', 'X_d3']
    # df_full is used for fitting (was for the DNN before, now is for the baselines)
    df_train = df_train.dropna()

    for df in [df_train, df_test]:
        print(df['mt'].value_counts())

    
    outcome = 'mt'
    outcome_def = 'mt'
    outcome_binary = 'mt'
    # print('dfs.keys())
    ############ finish getting data data ######################

    dataset_level_res = defaultdict(list)
    cell_level_res = defaultdict(list)
    models = []
    np.random.seed(42)
    
    
    # print("computing predictions for gb + rf + svm")
    for model_type in ['gb', 'rf', 'ridge', 'svm']:
        
        if model_type == 'rf':
            m = RandomForestRegressor(n_estimators=100, random_state=1)
        elif model_type == 'dt':
            m = DecisionTreeRegressor()
        elif model_type == 'linear':
            m = LinearRegression()
        elif model_type == 'ridge':
            m = RidgeCV()
        elif model_type == 'svm':
            m = SVR(gamma='scale')
        elif model_type == 'gb':
            m = GradientBoostingRegressor(random_state=1)
            
        for feat_set in ['basic', 'dasc']:
            models.append(f'{model_type}_{feat_set}')
            if feat_set == 'basic':
                feat_set = feat_names[1:]
            elif feat_set == 'dasc':
                feat_set = ['X_d1', 'X_d2', 'X_d3']
            
            
            # print('feat_set', feat_set)
            X_fit = df_train[feat_set]
            # print('X_fit shape', X_fit.shape)
            m.fit(X_fit, df_train[outcome_def].values)
        
            # df = ds[(k, v)]
            #if k == 'clath_aux+gak_a7d2_new':
            #    df = df.dropna()
            X = df_test[feat_set]
            X = X.fillna(X.mean())
            #y = df['Y_sig_mean_normalized']
            y_reg = df_test[outcome_def].values
            y = df_test[outcome_binary].values
            preds = m.predict(X)
            get_all_scores(y, preds, y_reg, df_test)                        

                    
                    
                    
    # print("computing predictions for lstm")                 
    models.append('lstm')
    checkpoint_fname = f'../models/vps_distingish_mt_vs_wt_epoch={epoch}.pkl'
    results = pkl.load(open(checkpoint_fname, 'rb'))
    dnn = neural_networks.neural_net_sklearn(
        D_in=40, H=20, p=0, arch='lstm', track_name=feat_name)
    dnn.model.load_state_dict(results['model_state_dict'])

    df = df_test
    X = df[[feat_name]]
    y_reg = df[outcome_def] # df['Y_sig_mean_normalized'].values
    y = df[outcome_binary].values
    #preds = np.logical_and(dnn.predict(X), df['X_max'] > 1500).values.astype(int)  
    preds = dnn.predict(X)
    print('shape', y.shape)
    print('means', (preds > 0.5).mean(), y.mean())
    print('acc', np.mean((preds > 0.5).astype(int) == y))
    get_all_scores(y, preds, y_reg, df)
                    
                    
    # print('saving')
    dataset_level_res = pd.DataFrame(dataset_level_res, index=models)
    dataset_level_res.to_csv(f"../reports/dataset_vpsmt_level_res_{outcome_binary}_epoch={epoch}.csv")
    
    # cell_level_res = pd.DataFrame(cell_level_res, index=models)
    # cell_level_res.to_csv(f"../reports/cell_vpsmt_level_res_{outcome_binary}_epoch={epoch}.csv")

