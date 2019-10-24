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
    
    return df

def preprocess(df):
    df['X_max'] = np.array([max(x) for x in df.X.values])
    df['Y_max'] = np.array([max(y) for y in df.Y.values])    
    return df