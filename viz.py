from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from os.path import join as oj
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
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
from sklearn.model_selection import KFold
from colorama import Fore
import pickle as pkl
from style import *
from sklearn.ensemble import IsolationForest
from sklearn import decomposition
from matplotlib_venn import venn3, venn2
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import matplotlib.gridspec as grd

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     fig, ax = plt.subplots()
    im = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    ax = plt.gca()
#     ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
#            title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax

def highlight_max(data, color='#0e5c99'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)
    
    
# visualize biggest errs
def viz_biggest_errs(df, idxs_cv, idxs, Y_test, preds, preds_proba, 
                     num_to_plot=20, aux_thresh=642):
    '''Visualize X and Y where the top examples are the most wrong / least confident
    Params
    ------
    idxs_cv: integer ndarray
        which idxs are not part of the test set (usually just 0, 1, 2, ...)
    idxs: boolean ndarray
        subset of points to plot
    
    '''
    
    # get args to sort by
    Y_test = Y_test[idxs]
    preds = preds[idxs]
    preds_proba = preds_proba[idxs]
    residuals = np.abs(Y_test - preds_proba)
    args = np.argsort(residuals)[::-1]
    
    dft = df.iloc[idxs_cv][idxs].iloc[args]
    lifetime_max = np.max(dft.lifetime.values)
    if num_to_plot is None:
        num_to_plot = dft.shape[0]
    R = int(np.sqrt(num_to_plot))
    C = num_to_plot // R + 1
    plt.figure(figsize=(C * 3, R * 2.5), dpi=200)
    
    i = 0
    for r in range(R):
        for c in range(C):
            if i < dft.shape[0]:
                ax = plt.subplot(R, C, i + 1)
                ax.text(.5, .9, f'{i}',
                         horizontalalignment='right',
                         transform=ax.transAxes)
                plt.axis('off')
                plt.plot(dft["X"].iloc[i], color=cr)
                plt.plot(dft["Y"].iloc[i], color='green')
                i += 1
                plt.xlim([-1, lifetime_max])
                plt.axhline(aux_thresh, color='gray', alpha=0.5)
            
    plt.tight_layout()
    return dft
    

def viz_errs_2d(df, idxs_test, preds, Y_test, key1='x_pos', key2='y_pos', X=None):
    '''visualize distribution of errs wrt to 2 dimensions
    '''
    
    if 'pc' in key1 and 'pc' in key2:
        pca = decomposition.PCA(n_components=2, whiten=True)
        X_reduced = pca.fit_transform(X.iloc[idxs_test])
        x_pos = X_reduced[:, 0]
        y_pos = X_reduced[:, 1]
    else:
        x_pos = df[key1].iloc[idxs_test]
        y_pos = df[key2].iloc[idxs_test]
    
    plt.figure(dpi=200)
    ms = 4
    me = 1
    plt.plot(x_pos[(preds==Y_test) & (preds==1)], y_pos[(preds==Y_test) & (preds==1)], 'o',
             color=cb, alpha=0.4, label='true pos', ms=ms, markeredgewidth=0)
    plt.plot(x_pos[(preds==Y_test) & (preds==0)], y_pos[(preds==Y_test) & (preds==0)], 'x',
             color=cb, alpha=0.4, label='true neg', ms=ms, markeredgewidth=1)
    plt.plot(x_pos[preds > Y_test], y_pos[preds > Y_test], 'o', color=cr, 
             alpha=0.4, label='false pos', ms=ms, markeredgewidth=0)    
    plt.plot(x_pos[preds < Y_test], y_pos[preds < Y_test], 'x', color=cr, 
             alpha=0.4, label='false neg', ms=ms, markeredgewidth=1)    
    plt.legend()
#     plt.scatter(x_pos, y_pos, c=preds==Y_test, alpha=0.5)
    plt.xlabel(key1)
    plt.ylabel(key2)
    plt.tight_layout()
    
    
def viz_errs_1d(X_test, preds, preds_proba, Y_test, norms, key='lifetime'):
    '''visualize errs based on lifetime
    '''
    plt.figure(dpi=200)
    correct_idxs = preds == Y_test
    lifetime = X_test[key] * norms[key]['std'] + norms[key]['mu']

    plt.plot(lifetime[(preds==Y_test) & (preds==1)], preds_proba[(preds==Y_test) & (preds==1)], 'o',
             color=cb, alpha=0.5, label='true pos')
    plt.plot(lifetime[(preds==Y_test) & (preds==0)], preds_proba[(preds==Y_test) & (preds==0)], 'x',
             color=cb, alpha=0.5, label='true neg')
    plt.plot(lifetime[preds > Y_test], preds_proba[preds > Y_test], 'o', color=cr, alpha=0.5, label='false pos')    
    plt.plot(lifetime[preds < Y_test], preds_proba[preds < Y_test], 'x', color=cr, alpha=0.5, label='false neg')    
    plt.xlabel(key)
    plt.ylabel('predicted probability')
    plt.legend()
    plt.show()
    
def plot_curves(df, extra_key=None, hline=True):
    '''Plot first  time-series curves from df
    '''
    plt.figure(figsize=(16, 10), dpi=200)
    R, C = 5, 8
    lifetime_max = np.max(df.lifetime.values[:R*C])
    for i in range(R * C):
        if i < df.shape[0]:
            plt.subplot(R, C, i + 1)
            row = df.iloc[i]
            plt.plot(row.X, color='red', label='clathrin')
            if extra_key is not None:
                plt.plot(row[extra_key], color='gray', label=extra_key)
            else:
                plt.plot(row.Y, color='green', label='auxilin')
                if hline:
                    plt.axhline(642.3754691658837, color='gray', alpha=0.5)
            plt.xlim([-1, lifetime_max + 1])
#         plt.ylim([-10, max(max(df.X_max), max(df.Y_max)) + 1])
    #     plt.axi('off')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def viz_errs_outliers_venn(X_test, preds, Y_test, num_feats_reduced=5):
    '''Compare outliers to errors in venn-diagram
    '''
    feat_names = data_tracks.get_feature_names(X_test)
    X_feat = X_test[feat_names]

    if num_feats_reduced is not None:
        pca = decomposition.PCA(n_components=num_feats_reduced)
        X_reduced = pca.fit_transform(X_feat)
    else:
        X_reduced = X_feat

    R, C = 2, 2
    titles = ['isolation forest', 'local outlier factor', 'elliptic envelop', 'one-class svm']
    plt.figure(figsize=(6, 5), dpi=200)
    for i in range(4):
        plt.subplot(R, C, i + 1)
        plt.title(titles[i])
        if i == 0:
            clf = IsolationForest(n_estimators=10, warm_start=True)
        elif i == 1:
            clf = LocalOutlierFactor(novelty=True)
        elif i == 2:
            clf = EllipticEnvelope()
        elif i == 3:
            clf = OneClassSVM()
        clf.fit(X_reduced)  # fit 10 trees  
        is_outlier = clf.predict(X_reduced)==-1
        is_err = preds != Y_test
        idxs = np.arange(is_outlier.size)
        venn2([set(idxs[is_outlier]), set(idxs[is_err])], set_labels=['outliers', 'errors'])
    
def plot_pcs(pca, X):
    '''Pretty plot of pcs with explained var bars
    Params
    ------
    pca: sklearn PCA class after being fitted
    '''
    plt.figure(figsize=(6, 9), dpi=200)
    
    # extract out relevant pars
    comps = pca.components_.transpose()
    var_norm = pca.explained_variance_ / np.sum(pca.explained_variance_) * 100
    
    
    # create a 2 X 2 grid 
    gs = grd.GridSpec(2, 2, height_ratios=[2,10], 
                      width_ratios=[12, 1], wspace=0.1, hspace=0)

    
    # plot explained variance
    ax2 = plt.subplot(gs[0])
    ax2.bar(np.arange(0, comps.shape[1]), var_norm, 
            color='gray', width=0.8)
    plt.title('Explained variance (%)')
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.set_yticks([0, max(var_norm)])
    plt.xlim((-0.5, comps.shape[1]-0.5))
    
    # plot pcs
    ax = plt.subplot(gs[2])
    vmaxabs = np.max(np.abs(comps))
    p = ax.imshow(comps, interpolation='None', aspect='auto',
                  cmap=sns.diverging_palette(10, 240, as_cmap=True, center='light'),
                  vmin=-vmaxabs, vmax=vmaxabs) # center at 0
    plt.xlabel('PCA component number')
    ax.set_yticklabels(list(X))
    ax.set_yticks(range(len(list(X))))
    

    # make colorbar
    colorAx = plt.subplot(gs[3])
    cb = plt.colorbar(p, cax=colorAx)
    plt.show()
    
    
def print_metadata(acc=None):
    metadata_file = 'processed/metadata.pkl'
    m = pkl.load(open(metadata_file, 'rb'))

    print(f'valid:\t\t{m["num_aux_pos_valid"]} aux+ / {m["num_tracks_valid"]} ({m["num_aux_pos_valid"]/m["num_tracks_valid"]:.3f})')
    print('----------------------------------------')
    print(f'no_hotspots:\t{m["num_aux_pos_after_hotspots"]} aux+ / {m["num_tracks_after_hotspots"]} ({m["num_aux_pos_after_hotspots"]/m["num_tracks_after_hotspots"]:.3f})')
    print('----------------------------------------')
    num_eval = m["num_tracks_after_hotspots"]
    
    if "num_aux_pos_early" in m:
        print(f'aux_early:\t{m["num_aux_pos_early"]:>4.0f} aux+ / {m["num_peaks_early"]:>4} ({m["num_aux_pos_early"]/m["num_peaks_early"]:.3f})')
        print(f'aux_late:\t{m["num_aux_pos_late"]:>4.0f} aux+ / {m["num_peaks_late"]:>4} ({m["num_aux_pos_late"]/m["num_peaks_late"]:.3f})')
        print(f'aux_valid:\t{m["num_aux_pos_after_peak_time"]:>4.0f} aux+ / {m["num_tracks_after_peak_time"]} ({m["num_aux_pos_after_peak_time"]/m["num_tracks_after_peak_time"]:.3f})')
        num_eval = m["num_tracks_after_peak_time"]
        print('----------------------------------------')

    print(f'lifetime<={m["thresh_short"]}:\t{round(m["num_short"] * m["acc_short"]):>4.0f} aux+ / {m["num_short"]:>4} ({m["acc_short"]:.3f})')
    print(f'lifetime>={m["thresh_long"]}:\t{round(m["num_long"] * m["acc_long"]):>4.0f} aux- / {m["num_long"]:>4} ({m["acc_long"]:.3f})')
    print(f'remaining:\t{m["num_aux_pos_after_lifetime"]:>4.0f} aux+ / {m["num_tracks_after_lifetime"]:>4} ({m["num_aux_pos_after_lifetime"]/m["num_tracks_after_lifetime"]:.3f})')
    if acc is not None:
        print('----------------------------------------')
        print(f'predicted acc:\t\t\t  {acc:.3f}')
        print(f'total acc:\t\t\t  {(m["num_short"] * m["acc_short"] + m["num_long"] * m["acc_long"] + acc * m["num_tracks_after_lifetime"]) / num_eval:.3f}')