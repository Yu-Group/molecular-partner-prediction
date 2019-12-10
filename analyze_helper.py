from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from os.path import join as oj
from sklearn.feature_extraction.image import extract_patches_2d
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
import models
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from colorama import Fore
import pickle as pkl
import viz
from style import *

def load_results(out_dir):
    r = []
    for fname in os.listdir(out_dir):
        d = pkl.load(open(oj(out_dir, fname), 'rb'))
        metrics = {k: d['cv'][k] for k in d['cv'].keys() if not 'curve' in k}
        out = {k: np.mean(metrics[k]) for k in metrics}
        out.update({k + '_std': np.std(metrics[k]) for k in metrics})
        out['model_type'] = fname.replace('.pkl', '') #d['model_type']
        
        imp_mat = np.array(d['imps']['imps'])
        imp_mu = imp_mat.mean(axis=0)
        imp_sd = imp_mat.std(axis=0)
        
        feat_names = d['feat_names_selected']
        out.update({feat_names[i] + '_f': imp_mu[i] for i in range(len(feat_names))})
        out.update({feat_names[i]+'_std_f': imp_sd[i] for i in range(len(feat_names))})
        r.append(pd.Series(out))
    r = pd.concat(r, axis=1, sort=False).T.infer_objects()
    r = r.reindex(sorted(r.columns), axis=1) # sort the column names
    r = r.round(3)
    r = r.set_index('model_type')
    return r

def analyze_individual_results(results, X_test, Y_test, print_results=False, plot_results=False, model_cv_fold=0):
    scores_cv = results['cv']
    scores_test = results['test']
    imps = results['imps']
    m = imps['model'][model_cv_fold]
    
    
    preds = m.predict(X_test[results['feat_names']])
    try:
        preds_proba = m.predict_proba(X_test[results['feat_names']])[:, 1]
    except:
        preds_proba = preds
    
    if print_results:
        print(Fore.CYAN + f'{"metric":<25}\tvalidation') #\ttest')
        for s in results['metrics']:
            if not 'curve' in s:
                print(Fore.WHITE + f'{s:<25}\t{np.mean(scores_cv[s]):.3f} ~ {np.std(scores_cv[s]):.3f}')
        #         print(Fore.WHITE + f'{s:<25}\t{np.mean(scores_cv[s]):.3f} ~ {np.std(scores_cv[s]):.3f}\t{np.mean(scores_test[s]):.3f} ~ {np.std(scores_test[s]):.3f}')

        print(Fore.CYAN + '\nfeature importances')
        imp_mat = np.array(imps['imps'])
        imp_mu = imp_mat.mean(axis=0)
        imp_sd = imp_mat.std(axis=0)
        for i, feat_name in enumerate(results['feat_names']):
            print(Fore.WHITE + f'{feat_name:<25}\t{imp_mu[i]:.3f} ~ {imp_sd[i]:.3f}')

    if plot_results:
        # print(m.coef_)
        plt.figure(figsize=(10, 3), dpi=140)
        R, C = 1, 3
        plt.subplot(R, C, 1)
        # print(X_test.shape, results['feat_names'])

        viz.plot_confusion_matrix(Y_test, preds, classes=np.array(['Failure', 'Success']))

        plt.subplot(R, C, 2)
        prec, rec, thresh = scores_test['precision_recall_curve'][0]
        plt.plot(rec, prec)
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.ylabel('Precision')
        plt.xlabel('Recall')


        plt.subplot(R, C, 3)
        plt.hist(preds_proba[Y_test==0], alpha=0.5, label='Failure')
        plt.hist(preds_proba[Y_test==1], alpha=0.5, label='Success')
        plt.xlabel('Predicted probability')
        plt.ylabel('Count')
        plt.legend()

        plt.tight_layout()
        plt.show()
    
    return preds, preds_proba