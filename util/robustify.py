from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from os.path import join as oj
from skimage.external.tifffile import imread
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.decomposition import sparse_encode
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
import pickle as pkl
from style import *
import math

def robust_measure(df, func):
    original_feature = np.array([func(df.iloc[i].X) for i in range(len(df))])
    res = []
    for noise_level in range(1, 500, 20):
        new_feature = np.array([func(df.iloc[i].X + 
                                     noise_level*np.random.normal(size=len(df.iloc[i].X))) 
                                for i in range(len(df))])
        res.append(np.corrcoef(np.transpose(original_feature), np.transpose(new_feature))[0,1])
    return res

