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
import data
from skorch.callbacks import Checkpoint
from skorch import NeuralNetRegressor
from config import *
from tqdm import tqdm
import pickle as pkl
import train_reg
from copy import deepcopy
import config
import models
import pandas as pd
import features
from scipy.stats import skew, pearsonr
import outcomes
import neural_networks
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.nn import functional as F
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix

# prepare data
dsets = ['clath_aux+gak_a7d2', 'clath_aux+gak', 'clath_aux+gak_a7d2_new', 'clath_aux+gak_new', 'clath_gak']
splits = ['train', 'test']
#feat_names = ['X_same_length_normalized'] + data.select_final_feats(data.get_feature_names(df))
              #['mean_total_displacement', 'mean_square_displacement', 'lifetime']
meta = ['cell_num', 'Y_sig_mean', 'Y_sig_mean_normalized']
dfs = data.load_dfs_for_lstm(dsets=dsets, splits=splits, meta=meta)


# select data for training
# df_train = dfs[('clath_aux+gak_a7d2_new', 'train')]
print(os.listdir(DIR_RESULTS))
df_full = pd.concat([dfs[(k, s)]
                     for (k, s) in dfs
                     if s == 'train'])[feat_names + ['Y_sig_mean_normalized', 'y_consec_sig']]
df_train = df_full.dropna()



# where to save
out_dir = f'{DIR_RESULTS}/dec13_deep_best'
# out_dir = f'results/regression/deep_learning/Dec11'


# actually train model
os.makedirs(out_dir, exist_ok=True)
outcome_def = 'Y_sig_mean_normalized'
num_epochs = 100
num_hidden = 40
for model_type in ['nn_lstm']: #['nn_cnn', 'fcnn', 'nn_lstm']: # = 'nn_cnn' # 'nn_lstm', 'fcnn', 'nn_cnn', 'nn_attention'
    train_reg.train_reg(df_train,
                        feat_names=feat_names,
                        track_name='X_same_length_normalized',
                        model_type=model_type, 
                        outcome_def=outcome_def,
                        out_name=oj(out_dir, f'{dset}_{outcome_def}_{model_type}_{num_epochs}_{num_hidden}.pkl'),
                        fcnn_hidden_neurons=num_hidden,
                        fcnn_epochs=num_epochs)
    
    
    
