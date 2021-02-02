import os
from os.path import join as oj
import sys
sys.path.append('../src')
import numpy as np
import torch
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

if __name__ == '__main__':
    
    dsets = ['clath_aux+gak_a7d2', 'clath_aux+gak', 'clath_aux+gak_a7d2_new', 'clath_aux+gak_new', 'clath_gak']
    splits = ['train', 'test']
    #feat_names = ['X_same_length_normalized'] + data.select_final_feats(data.get_feature_names(df))
                  #['mean_total_displacement', 'mean_square_displacement', 'lifetime']
    meta = ['cell_num', 'Y_sig_mean', 'Y_sig_mean_normalized']
    dfs = {}
    for dset in tqdm(dsets):
        for split in splits:
            df = data.get_data(dset=dset)
            df = df[~(df.short | df.long | df.hotspots)]
    #         df = df[df.valid]
            df = df[df.lifetime > 15] # only keep hard tracks
            df = df[df.cell_num.isin(config.DSETS[dset][split])] # exclude held-out test data
            feat_names = ['X_same_length_normalized'] + data.select_final_feats(data.get_feature_names(df))

            # downsample tracks
            length = 40
            df['X_same_length'] = [features.downsample(df.iloc[i]['X'], length)
                                   for i in range(len(df))] # downsampling
            # normalize tracks
            df = features.normalize_track(df, track='X_same_length', by_time_point=False)

            # regression response
            df = train_reg.add_sig_mean(df)     

            # remove extraneous feats
            # df = df[feat_names + meta]
    #         df = df.dropna() 

            # normalize features
            for feat in feat_names:
                if 'X_same_length' not in feat:
                    df = features.normalize_feature(df, feat)

            dfs[(dset, split)] = deepcopy(df)

    df_full = pd.concat([dfs[(k, s)]
                     for (k, s) in dfs
                     if s == 'train'])[feat_names + ['Y_sig_mean_normalized', 'y_consec_sig', 'y_consec_thresh']]
    df_full = df_full.dropna()
    ds = {(k, v): dfs[(k, v)]
          for (k, v) in sorted(dfs.keys(), key=lambda x: x[1] + x[0])
          #if not k == 'clath_aux+gak_a7d2_new'
         }
    corr_res = defaultdict(list)
    models = []
    np.random.seed(42)
    
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
            
            m.fit(df_full[feat_set], df_full['Y_sig_mean_normalized'].values)
        
            for i, (k, v) in enumerate(ds.keys()):
                if v == 'test':
                    df = ds[(k, v)]
                    if k == 'clath_aux+gak_a7d2_new':
                        df = df.dropna()
                    X = df[feat_set]
                    #y = df['Y_sig_mean_normalized']
                    y = df['y_consec_thresh']
                    preds = m.predict(X)
                    acc = np.mean((preds > 0) == y)
                    #acc = np.corrcoef(preds, y)[0, 1]
                    corr_res[k].append(100*acc)
                    
    models.append('lstm')
    results = pkl.load(open('../models/dnn_full_long_normalized_across_track_1_feat.pkl', 'rb'))
# results = pkl.load(open('../models/clath_aux+gak_a7d2_new_Y_sig_mean_normalized_nn_lstm_100_40.pkl', 'rb'))
    dnn = neural_networks.neural_net_sklearn(D_in=40, H=20, p=0, arch='lstm')
    dnn.model.load_state_dict(results['model_state_dict'])
    for i, (k, v) in enumerate(ds.keys()):
        if v == 'test':
            print(k)
            df = ds[(k, v)]
            X = df[feat_names[:1]]
                    #y = df['Y_sig_mean_normalized']
            y = df['y_consec_thresh'].values
            preds = dnn.predict(X)
            acc = np.mean((preds > 0).astype(int) == y)
            #acc = np.corrcoef(preds, y)[0, 1]
            corr_res[k].append(100*acc)    
                    
    corr_res = pd.DataFrame(corr_res, index=models)
    corr_res.to_csv("../reports/classification_results.csv")
                #print(k, v, acc)
                #plt.title(f'{k} {v} {100*acc:0.1f}', fontsize=10)    
