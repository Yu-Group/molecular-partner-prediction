import os
from os.path import join as oj
import sys
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
import pickle as pkl

if __name__ == '__main__':
    
    outcome_def = 'successful_full'
    print("loading data")
    dsets = ['clath_aux_dynamin']
    splits = ['test']
    #feat_names = ['X_same_length_normalized'] + data.select_final_feats(data.get_feature_names(df))
                  #['mean_total_displacement', 'mean_square_displacement', 'lifetime']
    meta = ['cell_num', 'Y_sig_mean', 'Y_sig_mean_normalized', outcome_def]
    dfs, feat_names = data.load_dfs_for_lstm(dsets=dsets, 
                                             splits=splits, 
                                             meta=meta,
                                             length=40,
                                             padding='end')
    df_test = pd.concat([dfs[(k, s)]
                                 for (k, s) in dfs
                                 if s == 'test'])[feat_names + meta]
    #df_test = df_test.dropna()
    X1 = df_test[feat_names[:1]]
    X2 = df_test[feat_names[1:]]
    X2 = X2.fillna(X2.mean())
    y = df_test[outcome_def].values
    
    accuracy = {}
    
    for k in [1, 2, 5, 10]:
        for j in tqdm(range(10)):
            
            checkpoint_fname = f'../models/models_different_size_10/downsample_{k}_batch_{j}_lstm.pkl'
            results = pkl.load(open(checkpoint_fname, 'rb'))
            dnn = neural_networks.neural_net_sklearn(D_in=40, H=20, p=0, arch='lstm', epochs=200)
            dnn.model.load_state_dict(results['model_state_dict'])
            preds = dnn.predict(X1)
            accuracy[(k, j, 'lstm')] = np.mean(y == (preds > 0))  

            checkpoint_fname = f'../models/models_different_size_10/downsample_{k}_batch_{j}_gb.pkl'
            m = pkl.load(open(checkpoint_fname, 'rb'))
            preds = m.predict(X2)
            accuracy[(k, j, 'gb')] = np.mean(y == (preds > 0))   
    pkl.dump(accuracy, open(f'../reports/data_size_stability_10_{outcome_def}.pkl', 'wb'))
    
    
    # calculate dasc accuracy
    dasc_pred = (df_test['X_d1'].values > 0).astype(int)
    dasc_acc = np.mean(y == dasc_pred)
    pkl.dump(dasc_acc, open('../reports/data_size_stability_10_dasc_acc.pkl', 'wb'))
            
