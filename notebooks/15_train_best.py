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
    #feat_names = [''] + data.select_final_feats(data.get_feature_names(df))
                  #['mean_total_displacement', 'mean_square_displacement', 'lifetime']
    length = 40
    padding = 'end'
    feat_name = 'X_same_length_extended_normalized' # include buffer X_same_length_normalized
    outcome = 'Y_sig_mean_normalized'
    for lifetime_threshold in [5, 10, 15]:
        dfs, feat_names = data.load_dfs_for_lstm(dsets=dsets, 
                                                 splits=splits, 
                                                 lifetime_threshold=lifetime_threshold,
                                                 length=length,
                                                 padding=padding)
        df_full = pd.concat([dfs[(k, s)]
                             for (k, s) in dfs
                             if s == 'train'])
        np.random.seed(42)
        checkpoint_fname = f'../models/dnn_fit_extended_lifetimes>{lifetime_threshold}.pkl'
        dnn = neural_networks.neural_net_sklearn(D_in=length, H=20, p=0, arch='lstm', epochs=200, track_name=feat_name)
        dnn.fit(df_full[[feat_name]],
                df_full[outcome].values,
                verbose=True, checkpoint_fname=checkpoint_fname, device='cuda')
        pkl.dump({'model_state_dict': dnn.model.cpu().state_dict()}, open(checkpoint_fname, 'wb'))