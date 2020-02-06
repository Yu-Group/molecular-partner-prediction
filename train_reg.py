from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from os.path import join as oj
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, train_test_split
from sklearn import metrics
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F
import torch
from copy import deepcopy
from sklearn import metrics
plt.style.use('dark_background')
import mat4py
import pandas as pd
import data_tracks
from skorch.callbacks import Checkpoint, TrainEndCheckpoint
from skorch import NeuralNetRegressor, NeuralNetClassifier
import models
from imblearn.over_sampling import RandomOverSampler
from statsmodels import robust
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import KFold
import pickle as pkl
from tqdm import tqdm
from sklearn.metrics import r2_score

cell_nums_train = np.array([1, 2, 3, 4, 5])
cell_nums_test = np.array([6])


def add_robust_features(df):
    df['X_95_quantile'] = np.array([np.quantile(df.iloc[i].X, 0.95) for i in range(len(df))])
    df['X_mad'] = np.array([robust.mad(df.iloc[i].X) for i in range(len(df))])
    return df

def log_transforms(df):
    df['X_max_log'] = np.log(df['X_max'])
    df['X_95_quantile_log'] = np.log(df['X_95_quantile'] + 1)
    df['Y_max_log'] = np.log(df['Y_max'])
    df['X_mad_log'] = np.log(df['X_mad'])
    def calc_rise_log(x):
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        rise = np.log(val_max) - np.log(abs(np.min(x[:idx_max + 1])) + 1) # max change before peak
        return rise

    def calc_fall_log(x):
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        fall = np.log(val_max) - np.log(abs(np.min(x[idx_max:])) + 1) # drop after peak
        return fall
    
    df['rise_log'] = np.array([calc_rise_log(df.iloc[i].X) for i in range(len(df))])
    df['fall_log'] = np.array([calc_fall_log(df.iloc[i].X) for i in range(len(df))])
    num = 3
    df['rise_local_3_log'] = df.apply(lambda row: 
                                  calc_rise_log(np.array(row['X'][max(0, row['X_peak_idx'] - num): 
                                                            row['X_peak_idx'] + num + 1])), 
                                          axis=1)
    df['fall_local_3_log'] = df.apply(lambda row: 
                                  calc_fall_log(np.array(row['X'][max(0, row['X_peak_idx'] - num): 
                                                            row['X_peak_idx'] + num + 1])), 
                                          axis=1)
    
    num2 = 11
    df['rise_local_11_log'] = df.apply(lambda row: 
                                   calc_rise_log(np.array(row['X'][max(0, row['X_peak_idx'] - num2): 
                                                            row['X_peak_idx'] + num2 + 1])), 
                                          axis=1)
    df['fall_local_11_log'] = df.apply(lambda row: 
                                   calc_fall_log(np.array(row['X'][max(0, row['X_peak_idx'] - num2): 
                                                            row['X_peak_idx'] + num2 + 1])), 
                                        axis=1)
    df['patch_diff_log'] = np.log(df['center_max']) - 1/4*(np.log(df['left_max'])
                                                          +np.log(df['right_max'])
                                                          +np.log(df['up_max'])
                                                          +np.log(df['down_max']))
    return df


def train_reg(df, feat_names, model_type='rf', outcome_def='Y_max_log',
              out_name='results/regression/test.pkl', seed=42):
    '''Run training and fit models
    This will balance the data
    This will normalize the features before fitting
    
    Params
    ------
    normalize: bool
        if True, will normalize features before fitting
    '''
    np.random.seed(seed)
    X = df[feat_names]
    #X = (X - X.mean()) / X.std() # normalize the data
    y = df[outcome_def].values


    if model_type == 'rf':
        m = RandomForestRegressor(n_estimators=100)
    elif model_type == 'dt':
        m = DecisionTreeRegressor()
    elif model_type == 'linear':
        m = LinearRegression()
    elif model_type == 'svm':
        m = SVR(gamma='scale')
    elif model_type == 'mlp2':
        m = MLPRegressor(hidden_layer_sizes=(10, ), max_iter=2000)
    elif model_type == 'gb':
        m = GradientBoostingRegressor()
    elif model_type == 'irf':
        m = irf.ensemble.wrf()
 
    #scores_cv = {s: [] for s in scorers.keys()}
    #scores_test = {s: [] for s in scorers.keys()}
    imps = {'model': [], 'imps': []}

    kf = KFold(n_splits=len(cell_nums_train))
    
    
    # split testing data based on cell num
    idxs_test = df.cell_num.isin(cell_nums_test)
    idxs_train = df.cell_num.isin(cell_nums_train)
    X_test, Y_test = X[idxs_test], y[idxs_test]
    num_pts_by_fold_cv = []
    y_preds = {}
    cv_score = []
    
    # loops over cv, where test set order is cell_nums_train[0], ..., cell_nums_train[-1]
    for cv_idx, cv_val_idx in kf.split(cell_nums_train):
        # get sample indices
        idxs_cv = df.cell_num.isin(cell_nums_train[np.array(cv_idx)])
        idxs_val_cv = df.cell_num.isin(cell_nums_train[np.array(cv_val_idx)])
        X_train_cv, Y_train_cv = X[idxs_cv], y[idxs_cv]
        X_val_cv, Y_val_cv = X[idxs_val_cv], y[idxs_val_cv]
        num_pts_by_fold_cv.append(X_val_cv.shape[0])
        
        # resample training data
        

        # fit
        m.fit(X_train_cv, Y_train_cv)

        # get preds
        preds = m.predict(X_val_cv)
        y_preds[cell_nums_train[np.array(cv_val_idx)][0]] = preds
        if 'log' in outcome_def:
            cv_score.append(r2_score(np.exp(Y_val_cv), np.exp(preds)))
        else:
            cv_score.append(m.score(X_val_cv, Y_val_cv))
    
    #cv_score = cv_score/len(cell_nums_train)
    m.fit(X[idxs_train], y[idxs_train])
    print(cv_score)
    test_preds = m.predict(X_test)
    results = {'y_preds': y_preds, 
               'y': y[idxs_train],
               'test_preds': test_preds,
               'cv': {'r2':cv_score}, 
               'model_type': model_type,
               'num_pts_by_fold_cv': np.array(num_pts_by_fold_cv),
              }
    # save results
    # os.makedirs(out_dir, exist_ok=True)

    pkl.dump(results, open(out_name, 'wb'))
    
    
def load_results(out_dir, by_cell=True):
    r = []
    for fname in os.listdir(out_dir):
        d = pkl.load(open(oj(out_dir, fname), 'rb'))
        metrics = {k: d['cv'][k] for k in d['cv'].keys() if not 'curve' in k}
        num_pts_by_fold_cv = d['num_pts_by_fold_cv']
        out = {k: np.average(metrics[k], weights=num_pts_by_fold_cv) for k in metrics}
        if by_cell:
            out.update({'cv_accuracy_by_cell': metrics['r2']})
        out.update({k + '_std': np.std(metrics[k]) for k in metrics})
        out['model_type'] = fname.replace('.pkl', '') #d['model_type']
        
        
        #imp_mat = np.array(d['imps']['imps'])
        #imp_mu = imp_mat.mean(axis=0)
        #imp_sd = imp_mat.std(axis=0)
        
        #feat_names = d['feat_names_selected']
        #out.update({feat_names[i] + '_f': imp_mu[i] for i in range(len(feat_names))})
        #out.update({feat_names[i]+'_std_f': imp_sd[i] for i in range(len(feat_names))})
        r.append(pd.Series(out))
    r = pd.concat(r, axis=1, sort=False).T.infer_objects()
    r = r.reindex(sorted(r.columns), axis=1) # sort the column names
    r = r.round(3)
    r = r.set_index('model_type')
    return r

