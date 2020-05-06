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
from sklearn import decomposition
import viz
import config

if __name__ == '__main__':
    SPLIT = config.SPLITS['orig']
    df = data.get_data(use_processed=True)
    df = df[df['valid']] # exclude test cells, short/long tracks, hotspots
    feat_names = data.get_feature_names(df)

    # don't use dict_feats
    feat_names = [x for x in feat_names 
                  if not x.startswith('sc_') 
                  and not x.startswith('nmf_')
                  and not x in ['center_max', 'left_max', 'right_max', 'up_max', 'down_max', 
                                'X_max_around_Y_peak', 'X_max_after_Y_peak', 'X_max_diff_after_Y_peak']
                  and not x.startswith('pc_')
    #               and not 'local' in x
    #               and not 'X_peak' in x
    #               and not 'slope' in x
    #               and not x in ['fall_final', 'fall_slope', 'fall_imp', 'fall']
                 ]
    feat_names = [x for x in feat_names if not '_tf_smooth' in x]
    feat_names = [x for x in feat_names if not 'local' in x]
    feat_names = [x for x in feat_names if not 'last' in x]
    # feat_names = [x for x in feat_names if '_tf_smooth' in x]
    print(feat_names)
    print('num feats', len(feat_names))



    outcome_def = 'y_consec_thresh'
    out_dir = oj('/scratch/users/vision/abc', 'apr28_1')
    os.makedirs(out_dir, exist_ok=True)
    feature_selection_nums = [3, 5, 15] #[3, 5, 7, 12, 16]: # number of feature to select [4, 9, 11, 23, 35, 39]
    for calibrated in [True, False]:
        for feature_selection_num in feature_selection_nums:
            for feature_selection in ['select_rf', None]: # select_lasso, select_rf, None
                if feature_selection is None and feature_selection_num > feature_selection_nums[0]: # don't do extra computation
                    break
                for model_type in tqdm(['logistic', 'rf', 'mlp2', 'svm']): #,'gb', 'logistic', 'dt', 'svm', 'gb', 'rf', 'mlp2', 'irf']):
                    for num_feats in [16, 25, len(feat_names)]: #[16, 25, len(feat_names)]: # number of total features to consider
                        for balancing in ['ros']: # None, ros                        
                            for balancing_ratio in [0.8, 1, 1.2]: # positive: negative  
                                hyperparams = [0] if model_type in ['logistic', 'rf', 'gb', 'dt', 'irf', 'qda'] else [-1, 0, 1]
                                for hyperparam in hyperparams: # 0 is default
                                    feats = feat_names[:num_feats]
                                    out_name = f'{model_type}_{num_feats}_{feature_selection}={feature_selection_num}_{balancing}={balancing_ratio}_h={hyperparam}_cal={calibrated}'
                                    train.train(df, feat_names=feats,
                                                cell_nums_feature_selection=SPLIT['feature_selection'],
                                                cell_nums_train=SPLIT['train'],
                                                model_type=model_type, 
                                                balancing=balancing, balancing_ratio=balancing_ratio,
                                                outcome_def=outcome_def,
                                                feature_selection=feature_selection,
                                                feature_selection_num=feature_selection_num,
                                                hyperparam=hyperparam,
                                                out_name=f'{out_dir}/{out_name}.pkl',
                                                calibrated=calibrated)