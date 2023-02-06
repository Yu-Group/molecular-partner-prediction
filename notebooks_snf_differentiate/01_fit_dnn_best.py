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
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    print("loading data")

    


    ############ get data ######################
    df_train, df_test, _ = data.get_snf_mt_vs_wt()
    # splits = ['train', 'test']
    length = 40
    #     padding = 'end'
    # feat_name = 'X_same_length_extended' # include buffer X_same_length_normalized
    feat_name = 'X_same_length' # include buffer X_same_length_normalized
    # outcome = 'Y_sig_mean_normalized'
    outcome = 'mt'
    epoch = 100

    df_full = df_train
    print('before dropping', df_full.shape)
    df_full = df_full[[feat_name, outcome]].dropna()
    print('after dropping', df_full.shape)
    print('vals', df_full['mt'].value_counts()) # 1791 class 0, 653 class 1
    
    ############ finish getting data data ######################

    np.random.seed(42)
    # checkpoint_fname = f'../models/dnn_vps_fit_extended_lifetimes>{lifetime_threshold}.pkl'
    checkpoint_fname = f'../models/vps_distingish_mt_vs_wt_epoch={epoch}.pkl'
    dnn = neural_networks.neural_net_sklearn(
        D_in=length, H=20, p=0, arch='lstm', lr=0.0001,
        epochs=epoch, track_name=feat_name
    )
    print('track_name', vars(dnn))
    dnn.fit(df_full[[feat_name]],
            df_full[outcome].values,
            verbose=True,
            checkpoint_fname=checkpoint_fname, device='cpu')
    pkl.dump({'model_state_dict': dnn.model.cpu().state_dict()}, open(checkpoint_fname, 'wb'))