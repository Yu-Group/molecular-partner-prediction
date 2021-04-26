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
    
    
    print("loading data")
    dsets = ['clath_aux+gak_a7d2', 'clath_aux+gak', 'clath_aux+gak_a7d2_new', 'clath_aux+gak_new', 'clath_gak', 'clath_aux_dynamin']
    splits = ['train', 'test']
    #feat_names = ['X_same_length_normalized'] + data.select_final_feats(data.get_feature_names(df))
                  #['mean_total_displacement', 'mean_square_displacement', 'lifetime']
    meta = ['cell_num', 'Y_sig_mean', 'Y_sig_mean_normalized', 'y_consec_thresh']
    dfs, feat_names = data.load_dfs_for_lstm(dsets=dsets, 
                                             splits=splits, 
                                             meta=meta,
                                             length=40,
                                             padding='end')
    df_full = pd.concat([dfs[(k, s)]
                                 for (k, s) in dfs
                                 if s == 'train'])[feat_names + meta]
    for k in [2, 5, 10]:
        np.random.seed(42)
        index_list = np.arange(len(df_full))
        np.random.shuffle(index_list)
        size = int(len(df_full)/k)
        for j in tqdm(range(k)):
            
            use_index = index_list[np.arange(j * size, (j + 1)*size)]
            df_full_train = df_full.iloc[use_index]
            print(len(df_full_train))
            checkpoint_fname = f'../models/models_different_size/downsample_{k}_batch_{j}_gb.pkl'
            m = GradientBoostingRegressor(random_state=1)
            X = df_full_train[feat_names[1:]]
            X = X.fillna(X.mean())
            m.fit(X, df_full_train['Y_sig_mean_normalized'].values)
            pkl.dump(m, open(checkpoint_fname, 'wb'))
            
            checkpoint_fname = f'../models/models_different_size/downsample_{k}_batch_{j}_lstm.pkl'
            dnn = neural_networks.neural_net_sklearn(D_in=40, H=20, p=0, arch='lstm', epochs=200)
            dnn.fit(df_full_train[feat_names[:1]], df_full_train['Y_sig_mean_normalized'].values, verbose=True, checkpoint_fname=checkpoint_fname)
            pkl.dump({'model_state_dict': dnn.model.state_dict()}, open(checkpoint_fname, 'wb'))
            

            