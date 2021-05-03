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
    for length in [40, 100, 200]:
        for padding in ['front', 'end']:
            dfs, feat_names = data.load_dfs_for_lstm(dsets=dsets, 
                                                     splits=splits, 
                                                     meta=meta,
                                                     length=length,
                                                     padding=padding)
            df_full = pd.concat([dfs[(k, s)]
                                 for (k, s) in dfs
                                 if s == 'train'])[feat_names + meta]
            np.random.seed(42)
            checkpoint_fname = f'../models/dnn_full_long_normalized_across_track_1_feat_dynamin_{length}_{padding}_tuning.pkl'
            valid_cells = ['A7D2/1', 
                           'CLTA-TagRFP EGFP-Aux1 EGFP-GAK F6/1', 
                           'CLTA-TagRFP EGFP-GAK A8/1', 
                           'EGFP-GAK F6/1',
                           '488-1.5mW 561-1.5mW 647-1.5mW Exp100ms Int1.5s_4_Pos0/1_1.5s',
                           '488-1.5mW 561-1.5mW 647-1.5mW Exp100ms Int1.5s_4_Pos1/1_1.5s',
                           '488-1.5mW 561-1.5mW 647-1.5mW Exp100ms Int1.5s_4_Pos2/1_1.5s']
            valid = df_full['cell_num'].isin(valid_cells)
            df_full_train = df_full[~valid]
            dnn = neural_networks.neural_net_sklearn(D_in=length, H=20, p=0, arch='lstm', epochs=200)
            dnn.fit(df_full_train[feat_names[:1]], df_full_train['Y_sig_mean_normalized'].values, verbose=True, checkpoint_fname=checkpoint_fname)
            pkl.dump({'model_state_dict': dnn.model.state_dict()}, open(checkpoint_fname, 'wb'))