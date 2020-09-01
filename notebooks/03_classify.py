import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import os
from os.path import join as oj
plt.style.use('dark_background')
import data
from tqdm import tqdm
from util import train
from util.style import *
import config

if __name__ == '__main__':
    # some settings
    outcome_def = 'y_consec_thresh'
    out_dir = oj('/scratch/users/vision/chandan/abc', 'aug19_binary1')
    DSET = config.DSETS['clath_aux+gak_a7d2']
    binarize=True
    
    
    # get data
    df = data.get_data(use_processed=True)
    df = df[df['valid']] # exclude test cells, short/long tracks, hotspots
    feat_names = data.get_feature_names(df)
    feat_names = data.select_final_feats(feat_names, binarize=binarize)
    print(feat_names)
    print('num feats', len(feat_names))

    # run
    os.makedirs(out_dir, exist_ok=True)
    feature_selection_nums = [2, 3, 4, 5, 6, 7, 8, 9, 15, 100] #[3, 5, 7, 12, 16]: # number of feature to select [4, 9, 11, 23, 35, 39]
    for calibrated in [True, False]:
        for feature_selection_num in feature_selection_nums:
            for feature_selection in ['select_lasso', 'select_rf']: # select_lasso, select_rf, None
                if feature_selection is None and feature_selection_num > feature_selection_nums[0]: # don't do extra computation
                    break
                for model_type in tqdm(['logistic', 'rf', 'mlp2', 'svm']): #,'gb', 'logistic', 'dt', 'svm', 'gb', 'rf', 'mlp2', 'irf']):
                    for num_feats in [len(feat_names)]: # [16, 25, len(feat_names)]: #[16, 25, len(feat_names)]: # number of total features to consider
                        for balancing in ['ros']: # None, ros                        
                            for balancing_ratio in [0.8, 1, 1.2]: # positive: negative  
                                hyperparams = [0] if model_type in ['logistic', 'rf', 'gb', 'dt', 'irf', 'qda'] else [-1, 0, 1]
                                for hyperparam in hyperparams: # 0 is default
                                    feats = feat_names[:num_feats]
                                    out_name = f'{model_type}_{num_feats}_{feature_selection}={feature_selection_num}_{balancing}={balancing_ratio}_h={hyperparam}_cal={calibrated}'
                                    train.train(df, feat_names=feats,
                                                cell_nums_feature_selection=DSET['feature_selection'],
                                                cell_nums_train=DSET['train'],
                                                model_type=model_type, 
                                                balancing=balancing, balancing_ratio=balancing_ratio,
                                                outcome_def=outcome_def,
                                                feature_selection=feature_selection,
                                                feature_selection_num=feature_selection_num,
                                                hyperparam=hyperparam,
                                                out_name=f'{out_dir}/{out_name}.pkl',
                                                calibrated=calibrated)