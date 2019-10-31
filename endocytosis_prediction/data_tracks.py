from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from os.path import join as oj
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, train_test_split
import numpy as np
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F
import torch
from copy import deepcopy
from sklearn import metrics
plt.style.use('dark_background')
import mat4py
import pandas as pd

def get_tracks(fname = '/scratch/users/vision/data/abc_data/auxilin_data_tracked/A7D2/Cell1_1s/TagRFP/Tracking/ProcessedTracks.mat'):
    mat = mat4py.loadmat(fname)
    tracks = mat['tracks']
    n = len(tracks['t'])
    totalDisplacement = []
    for i in range(n):
        try:
            totalDisplacement.append(tracks['MotionAnalysis'][i]['totalDisplacement'])
        except:
            totalDisplacement.append(0)
    X = np.array([mat['tracks']['A'][i][0] for i in range(n)])
    Y = np.array([mat['tracks']['A'][i][1] for i in range(n)])
#     df = pd.DataFrame(tracks)
#     print(df.keys()) # these lines help us look at the other stored vars
    df = pd.DataFrame.from_dict({
        'X': X, 
        'Y': Y,
        'totalDisplacement': totalDisplacement
    })
    df['len'] = np.array([len(x) - np.sum(np.isnan(x)) for x in df.X.values])
    return df

def preprocess(df):
    df = df[df.len > 2]
    df['X_max'] = np.array([max(x) for x in df.X.values])
    df['X_mean'] = np.nan_to_num(np.array([np.nanmean(x) for x in df.X.values]))
    df['X_std'] = np.nan_to_num(np.array([np.std(x) for x in df.X.values]))
    df['Y_max'] = np.array([max(y) for y in df.Y.values])    
    df['Y_mean'] = np.nan_to_num(np.array([np.nanmean(y) for y in df.Y.values]))
    df['Y_std'] = np.nan_to_num(np.array([np.std(y) for y in df.Y.values]))
    return df

def add_outcome(df, thresh=3.25):
    '''Add binary outcome of whether spike happened and info on whether events were questionable
    '''
    df['outcome_score'] = df['Y_max'].values - (df['Y_mean'].values + thresh * df['Y_std'].values)
    df['outcome'] = (df['outcome_score'].values > 0).astype(np.int) # Y_max was big
    
    
    # multiple events (at the same location)
    df['multiple'] = None
    
    # partial events (signal is already high at the start)
    df['partial'] = None
    
    return df

def extract_X_mat(df, p=300):
    '''Extract matrix for X filled with zeros after sequences end
    '''
    n = df.shape[0]
    X_mat = np.zeros((n, p)).astype(np.float32)
    X = df['X'].values
    for i in range(n):
        x = X[i]
        num_timepoints = min(300, len(x))
        X_mat[i, :num_timepoints] = x[:num_timepoints]
    X_mat = np.nan_to_num(X_mat)
    X_mat -= np.min(X_mat)
    X_mat /= np.std(X_mat)
    return X_mat