from matplotlib import pyplot as plt
import os
from os.path import join as oj
import numpy as np
import pandas as pd
import data
from sklearn.model_selection import KFold
from colorama import Fore
import pickle as pkl
import config
import viz


def load_results(out_dir):
    r = []
    for fname in os.listdir(out_dir):
        d = pkl.load(open(oj(out_dir, fname), 'rb'))
        metrics = {k: d['cv'][k] for k in d['cv'].keys() if not 'curve' in k}
        num_pts_by_fold_cv = d['num_pts_by_fold_cv']
        out = {k: np.average(metrics[k], weights=num_pts_by_fold_cv) for k in metrics}
        out.update({k + '_std': np.std(metrics[k]) for k in metrics})
        out['model_type'] = fname.replace('.pkl', '')  # d['model_type']

        imp_mat = np.array(d['imps']['imps'])
        imp_mu = imp_mat.mean(axis=0)
        imp_sd = imp_mat.std(axis=0)

        feat_names = d['feat_names_selected']
        out.update({feat_names[i] + '_f': imp_mu[i] for i in range(len(feat_names))})
        out.update({feat_names[i] + '_std_f': imp_sd[i] for i in range(len(feat_names))})
        r.append(pd.Series(out))
    r = pd.concat(r, axis=1, sort=False).T.infer_objects()
    r = r.reindex(sorted(r.columns), axis=1)  # sort the column names
    r = r.round(3)
    r = r.set_index('model_type')
    return r


def get_data_over_folds(model_names: list, out_dir: str, cell_nums: pd.Series, X, y, outcome_def='y_consec_sig', dset='clath_aux+gak_a7d2'):
    '''Returns predictions/labels over folds in the dataset
    Params
    ------
    cell_nums: pd.Series
        equivalent to df.cell_num
    
    Returns
    -------
    d_full_cv: pd.DataFrame
        n rows, one for each data point in the training set (over all folds)
        2 columns for each model, one for predictions, and one for predicted probabilities
    idxs_cv: np.array
        indexes corresponding locations of the validation set
        for example, df.y_thresh.iloc[idxs_cv] would yield all the labels corresponding to the cv preds
    '''
    # split testing data based on cell num
    d = {}
    cell_nums_train = config.DSETS[dset]['train']
    kf = KFold(n_splits=len(cell_nums_train))
    idxs_cv = []

    # get predictions over all folds and save into a dict
    if not type(model_names) == 'list' and not 'ndarray' in str(type(model_names)):
        model_names = [model_names]
    for i, model_name in enumerate(model_names):
        results_individual = pkl.load(open(f'{out_dir}/{model_name}.pkl', 'rb'))

        fold_num = 0
        for cv_idx, cv_val_idx in kf.split(cell_nums_train):
            # get sample indices
            idxs_val_cv = cell_nums.isin(cell_nums_train[np.array(cv_val_idx)])
            X_val_cv, Y_val_cv = X[idxs_val_cv], y[idxs_val_cv]

            # get predictions
            preds, preds_proba = analyze_individual_results(results_individual, X_val_cv, Y_val_cv,
                                                            print_results=False, plot_results=False,
                                                            model_cv_fold=fold_num)

            d[f'{model_name}_{fold_num}'] = preds
            d[f'{model_name}_{fold_num}_proba'] = preds_proba

            if i == 0:
                idxs_cv.append(np.arange(X.shape[0])[idxs_val_cv])

            fold_num += 1

    # concatenate over folds
    d2 = {}
    for model_name in model_names:
        d2[model_name] = np.hstack([d[k] for k in d.keys() if model_name in k and not 'proba' in k])
        d2[model_name + '_proba'] = np.hstack([d[k] for k in d.keys() if model_name in k and 'proba' in k])
    return pd.DataFrame.from_dict(d2), np.hstack(idxs_cv)


def analyze_individual_results(results, X_test, Y_test, print_results=False, plot_results=False, model_cv_fold=0):
    scores_cv = results['cv']
    imps = results['imps']
    m = imps['model'][model_cv_fold]

    preds = m.predict(X_test[results['feat_names_selected']])
    try:
        preds_proba = m.predict_proba(X_test[results['feat_names_selected']])[:, 1]
    except:
        preds_proba = preds

    if print_results:
        print(Fore.CYAN + f'{"metric":<25}\tvalidation')  # \ttest')
        for s in results['metrics']:
            if not 'curve' in s:
                print(Fore.WHITE + f'{s:<25}\t{np.mean(scores_cv[s]):.3f} ~ {np.std(scores_cv[s]):.3f}')
        #         print(Fore.WHITE + f'{s:<25}\t{np.mean(scores_cv[s]):.3f} ~ {np.std(scores_cv[s]):.3f}\t{np.mean(scores_test[s]):.3f} ~ {np.std(scores_test[s]):.3f}')

        print(Fore.CYAN + '\nfeature importances')
        imp_mat = np.array(imps['imps'])
        imp_mu = imp_mat.mean(axis=0)
        imp_sd = imp_mat.std(axis=0)
        for i, feat_name in enumerate(results['feat_names_selected']):
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
        plt.hist(preds_proba[Y_test == 0], alpha=0.5, label='Failure')
        plt.hist(preds_proba[Y_test == 1], alpha=0.5, label='Success')
        plt.xlabel('Predicted probability')
        plt.ylabel('Count')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return preds, preds_proba


def load_results_many_models(out_dir, model_names, X_test, Y_test):
    d = {}
    for i, model_name in enumerate(model_names):
        results_individual = pkl.load(open(oj(out_dir, f'{model_name}.pkl'), 'rb'))
        preds, preds_proba = analyze_individual_results(results_individual, X_test, Y_test,
                                                        print_results=False, plot_results=False)
        d[model_name] = preds
        d[model_name + '_proba'] = preds_proba
        d[model_name + '_errs'] = preds != Y_test
    df_preds = pd.DataFrame.from_dict(d)
    return df_preds


# normalize and store
def normalize(df, outcome_def):
    X = df[data.get_feature_names(df)]
    X_mean = X.mean()
    X_std = X.std()
    ks = list(X.keys())
    norms = {ks[i]: {'mu': X_mean[i], 'std': X_std[i]} for i in range(len(ks))}
    X = (X - X_mean) / X_std
    y = df[outcome_def].values
    return X, y, norms


def calc_errs(preds, y_full_cv):
    tp = np.logical_and(preds == 1, y_full_cv == 1)
    tn = np.logical_and(preds == 0, y_full_cv == 0)
    fp = preds > y_full_cv
    fn = preds < y_full_cv
    return tp, tn, fp, fn
