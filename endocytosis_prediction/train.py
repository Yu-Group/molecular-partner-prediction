
import seaborn as sns
import numpy as np
import os
from os.path import join as oj
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import metrics
import eli5
import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from torch import nn
import torch.nn.functional as F
import torch
from copy import deepcopy
from sklearn import metrics

import mat4py
import pandas as pd
import data_tracks
from skorch.callbacks import Checkpoint, TrainEndCheckpoint
from skorch import NeuralNetRegressor, NeuralNetClassifier
import models
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import KFold
from colorama import Fore
import pickle as pkl
from tqdm import tqdm


def get_feature_importance(model, model_type, X_val, Y_val):
    if model_type in ['rf', 'dt']:
        imps = model.feature_importances_
    elif model_type == 'logistic':
        imps = model.coef_
    else:
        perm = eli5.sklearn.permutation_importance.PermutationImportance(model).fit(X_val, Y_val)
        imps = perm.feature_importances_
    return imps.squeeze()

def balance(X, y, balancing='ros'):
    '''
    Balance classes in y using strategy specified by balancing
    '''
    if balancing == 'ros':
        sampler = RandomOverSampler(random_state=42)
    elif balancing == 'smote':
        sampler = SMOTE(random_state=42)
    X_r, Y_r = sampler.fit_resample(X, y)   
    return X_r, Y_r
    
    

def train(df, feat_names, model_type='rf', outcome_def='y_thresh',
          balancing='ros', out_name='results/classify/test.pkl'):
    np.random.seed(42)
    # make logistic data
    X = df[feat_names]
    X = (X - X.mean()) / X.std()
    y = df[outcome_def].values

    # split testing data based on cell num
    idxs_test = df.cell_num.isin([6])
    X_test, Y_test = X[idxs_test], y[idxs_test]

    if model_type == 'rf':
        m = RandomForestClassifier(n_estimators=100)
    elif model_type == 'dt':
        m = DecisionTreeClassifier()
    elif model_type == 'logistic':
        m = LogisticRegression(solver='lbfgs')
    elif model_type == 'svm':
        m = SVC(gamma='scale')
    elif model_type == 'mlp2':
        m = MLPClassifier()
    elif model_type == 'gb':
        m = GradientBoostingClassifier()
    

    # scores = ['balanced_accuracy'] # ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy', 'roc_auc']
    scorers = {'balanced_accuracy': metrics.balanced_accuracy_score, 'accuracy': metrics.accuracy_score,
               'precision': metrics.precision_score, 'recall': metrics.recall_score, 'f1': metrics.f1_score, 'roc_auc': metrics.roc_auc_score,
               'precision_recall_curve': metrics.precision_recall_curve, 'roc_curve': metrics.roc_curve}
    scores_cv = {s: [] for s in scorers.keys()}
    scores_test = {s: [] for s in scorers.keys()}
    imps = {'model': [], 'imps': []}

    kf = KFold(n_splits=5)
    cell_nums_train = [1, 2, 3, 4, 5]
    for cv_idx, cv_val_idx in kf.split(cell_nums_train):
        # get sample indices
        idxs_cv = df.cell_num.isin(cv_idx + 1)
        idxs_val_cv = df.cell_num.isin(cv_val_idx + 1)
        X_train_cv, Y_train_cv = X[idxs_cv], y[idxs_cv]
        X_val_cv, Y_val_cv = X[idxs_val_cv], y[idxs_val_cv]


        # resample training data
        X_train_r_cv, Y_train_r_cv = balance(X_train_cv, Y_train_cv, balancing)

        # fit
        m.fit(X_train_r_cv, Y_train_r_cv)

        # get preds
        preds = m.predict(X_val_cv)
        preds_test = m.predict(X_test)
        if model_type == 'svm':
            preds_proba = preds
            preds_test_proba = preds_test
        else:
            preds_proba = m.predict_proba(X_val_cv)[:, 1]
            preds_test_proba = m.predict_proba(X_test)[:, 1]


        # add scores
        for s in scorers.keys():
            scorer = scorers[s]
            if 'roc' in s or 'curve' in s:
                scores_cv[s].append(scorer(Y_val_cv, preds_proba))
                scores_test[s].append(scorer(Y_test, preds_test_proba))
            else:
                scores_cv[s].append(scorer(preds, Y_val_cv))
                scores_test[s].append(scorer(preds_test, Y_test))
        imps['model'].append(deepcopy(m))
        imps['imps'].append(get_feature_importance(m, model_type, X_val_cv, Y_val_cv))

    # save results
    # os.makedirs(out_dir, exist_ok=True)
    results = {'metrics': list(scorers.keys()), 'cv': scores_cv, 
               'test': scores_test, 'imps': imps,
               'feat_names': feat_names,
               'model_type': model_type,
               'balancing': balancing,
              }
    pkl.dump(results, open(out_name, 'wb'))