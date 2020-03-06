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
pd.options.mode.chained_assignment = None  # default='warn' - caution: this turns off setting with copy warning
import pickle as pkl
from style import *
import math
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import DictionaryLearning, NMF
from sklearn import decomposition


auxilin_dir = '/scratch/users/vision/data/abc_data/auxilin_data_tracked'

# data splitting
cell_nums_feature_selection = np.array([1])
cell_nums_train = np.array([1, 2, 3, 4, 5])
cell_nums_test = np.array([6, 7, 8]) # currently these are not even loaded

def get_data(use_processed=True, save_processed=True, processed_file='processed/df.pkl', 
             metadata_file='processed/metadata.pkl', use_processed_dicts=True, 
             outcome_def='y_consec_thresh', all_data=False):
    '''
    Params
    ------
    use_processed: bool, optional
        determines whether to load df from cached pkl
    save_processed: bool, optional
        if not using processed, determines whether to save the df
    use_processed_dicts: bool, optional
        if False, recalculate the dictionary features
    '''
    
    if use_processed and os.path.exists(processed_file):
        return pd.read_pickle(processed_file)
    else:
        print('loading + preprocessing data...')
        metadata = {}
        print('\tloading tracks...')
        df = get_tracks(all_data=all_data) # note: different Xs can be different shapes
        df['pid'] = np.arange(df.shape[0]) # assign each track a unique id
        df['valid'] = 1 # all tracks start as valid
        df['valid'][df.cell_num.isin(cell_nums_test)] = 0
        metadata['num_tracks_orig'] = df.shape[0]
        
        print('\tpreprocessing data...')
        df = remove_invalid_tracks(df)
        df = preprocess(df)
        df = add_outcomes(df)
        metadata['num_tracks_valid'] = df.valid.sum()
        metadata['num_aux_pos_valid'] = df[df.valid == 1][outcome_def].sum()
        metadata['num_hospots_valid'] = df[df.valid == 1]['hotspots'].sum()
        
        
        df[df['hotspots'] == 1]['valid'] = 0
        metadata['num_tracks_after_hotspots'] = df['valid'].sum()
        metadata['num_aux_pos_after_hotspots'] = df[df.valid == 1][outcome_def].sum()
        
        df, meta_lifetime = remove_tracks_by_lifetime(df, outcome_def=outcome_def, plot=False, acc_thresh=0.92)
        metadata.update(meta_lifetime)
        
        print('\tadding features...')
        # df = add_dict_features(df, use_processed=use_processed_dicts)
        # df = add_smoothed_tracks(df)
        df = add_pcs(df)
        if save_processed:
            pkl.dump(metadata, open(metadata_file, 'wb'))
            df.to_pickle(processed_file)
    return df

# gt labels (by pid)
def get_labels():
    '''Note these labels constitute corrections to the y_consec_thresh defn (as of 02/09/2020)
    '''
    labels = {
        'neg': [6725, 909, 5926, 983, 8224, 3363] + [221, 248, 255, 274, 291, 294, 350, 358, 364, 384, 434, 481, 518, 535, 588, 590, 602, 604, 638, 668, 675, 708, 752, 778, 808, 827, 852, 867, 884, 888, 897, 905, 952, 953, 967, 984, 987, 997, 1002, 1003, 1015, 1030, 1045, 1051, 1055, 1070, 1091, 1100, 1144, 1178, 1226, 1246, 1258, 1260, 1276, 1307, 1317, 1370, 1394, 1399, 1400, 1423, 1432, 1451, 1613, 2115, 2143, 2286, 2330, 2369, 2381, 2399, 2416, 2435, 2612, 2613, 2631, 2667, 2679, 2705, 2714, 2750, 2756, 2857, 2962, 3002, 3017, 3032, 3061, 3128, 3181, 3216, 3801, 3819, 3916, 3945, 3954, 3967, 3970, 4042, 4108, 4120, 4141, 4142, 4155, 4162, 4165, 4172, 4221, 4279, 4290, 4297, 4328, 4333, 4341, 4384, 4422, 4428, 4468, 4547, 4558, 4573, 4586, 4616, 4621, 4674, 4875, 4936, 4941, 4947, 5151, 5391, 5602, 5693, 5706, 5707, 5946, 5973, 6007, 6072, 6147, 6157, 6253, 6489, 6618, 7340, 7363, 7413, 7423, 7424, 7455, 7483, 7485, 7512, 7565, 7568, 7641, 7647, 7690, 7728, 7872, 7877, 7883, 7892, 7942, 7969, 7985, 8005, 8018, 8020, 8024, 8111, 8120, 8166, 8255, 8264, 8375, 8462, 8484, 8512, 8532, 8829, 8833, 8918, 8944] + [3964, 10054, 735, 846, 1362, 3823, 5389, 8834, 9565, 10882, 11166, 718] + [3964, 718, 735, 846, 1362, 3823, 5389, 8834, 9565, 10882, 11166, 10054], # note: first list were flipped, second list were already correct
        'pos': [3982, 8243, 777, 3940, 7559, 2455, 4748, 633, 2177, 1205, 603, 7972, 8458, 3041, 924, 8786, 4116, 885, 6298, 4658, 7889, 982, 829, 1210, 3054, 504, 1164, 347, 627, 1470, 2662, 2813, 422, 8400, 7474, 1273, 6365, 1559, 4348, 1156, 6250, 4864, 639, 930, 5424, 7818, 8463, 4358, 7656, 843, 890, 4373, 2737, 7524, 2590, 3804, 7667, 2148, 8585, 2919, 5712, 4440, 1440, 4699, 1089, 3004, 3126, 2869, 4183, 7335, 3166, 8461, 2180, 849, 6458, 4575, 4091, 3966, 4725, 2514, 7626, 3055, 4200, 6429, 1220, 4472, 8559, 412, 903, 5440, 1084, 2136, 6833, 1189, 7521, 8141, 7939, 8421, 944, 1264, 298, 6600, 1309, 3043, 243, 4161, 6813, 5464] + [238, 251, 389, 524, 556, 758, 759, 830, 1213, 1255, 1290, 1422, 1463, 1484, 1504, 2016, 2046, 2061, 2077, 2083, 2106, 2112, 2116, 2124, 2129, 2135, 2195, 2209, 2217, 2234, 2273, 2291, 2313, 2338, 2353, 2402, 2460, 2651, 2658, 2703, 2791, 2805, 2848, 3101, 3138, 3142, 3150, 3170, 3343, 3367, 3918, 3946, 4223, 4386, 4430, 4583, 4608, 4620, 4724, 5366, 5383, 5384, 5407, 5412, 5415, 5422, 5442, 5447, 5449, 5478, 5492, 5503, 5506, 5516, 5548, 5550, 5558, 5589, 5634, 5694, 5708, 5728, 5760, 5780, 5787, 5788, 5800, 5811, 5813, 5814, 5879, 5882, 5885, 5888, 5891, 5899, 5911, 5912, 5950, 5951, 5953, 5957, 5960, 5986, 5988, 6000, 6011, 6012, 6020, 6021, 6032, 6049, 6053, 6065, 6096, 6106, 6113, 6118, 6123, 6152, 6155, 6202, 6237, 6246, 6248, 6263, 6266, 6272, 6273, 6302, 6305, 6321, 6325, 6327, 6363, 6368, 6398, 6407, 6410, 6423, 6424, 6431, 6444, 6449, 6461, 6462, 6482, 6490, 6517, 6518, 6526, 6568, 6586, 6594, 6601, 6608, 6640, 6656, 6662, 6683, 6684, 6693, 6703, 6771, 6774, 6801, 6802, 6823, 6851, 7348, 7352, 7448, 7470, 7496, 7511, 7596, 7720, 7787, 7805, 7819, 7826, 7885, 7900, 7908, 7926, 7930, 7951, 7965, 8000, 8072, 8109, 8122, 8123, 8143, 8159, 8211, 8242, 8248, 8257, 8259, 8265, 8286, 8321, 8330, 8357, 8368, 8372, 8385, 8407, 8430, 8436, 8444, 8448, 8454, 8490, 8507, 8513, 8556, 8604, 8639, 8750, 8751, 8755, 8764, 8777, 8822, 8852, 8863, 8911, 8981, 2668, 889, 6066, 9529, 9676, 9990, 10157, 10183, 10243, 10434] + [2321, 11032, 4484, 4750, 8084, 6770, 6624, 2749, 6378, 7833, 4399, 9547, 2253] + [2321, 10996, 4454, 10431, 11032, 10057, 4484, 8084, 10754, 2382, 938, 2228, 10887, 6770, 10895, 10863, 6624, 10333, 4069, 10113, 4849, 9719, 10116, 245, 10077, 9547, 9557, 10457, 10037, 9900, 10146, 5507, 10517, 2749, 9563, 6378, 2014, 9714, 1353, 10117, 7504, 9724, 3141, 5797, 10508, 10374, 5593, 9932, 4399, 10632, 1039, 9904, 9930, 8505, 429, 10331, 5470, 8557, 7773, 10830, 10749, 2031, 3822, 7833, 5791, 10602, 2203, 542, 10843, 7759, 10483, 4827, 225, 7679, 9617, 2378, 5409, 10142, 9975, 10264, 918, 10148, 10066, 9917, 9485, 6400, 5961, 10023, 10418, 231, 10695, 3065, 6420, 7865, 9813, 10765, 6290, 2270, 729, 2626, 8424, 10199, 2200, 2854, 2253], # note: first list were flipped, second list were already correct
    'hotspots': [6510, 6606, 2373, 6135, 6023, 7730, 2193, 8307, 5626, 4109, 2921, 4614, 2573, 7490, 6097, 7836, 1011, 6493, 5779, 8660, 6232, 6009, 2579, 929, 3824, 357, 6162, 477, 5640, 6467, 244, 2922, 4288, 2926, 1480, 4441, 4683, 8239, 9749, 9826, 9844, 10945, 11037] + [10069, 5485, 3146, 5560, 5600, 5937, 7688, 6055, 5670, 10235, 5583, 6151, 5720, 2553, 6040, 292, 5456, 2437, 5966, 5499, 10043, 10232, 5434, 6224, 5785, 6210, 2761, 6359, 6438, 5423, 5774, 7556, 5766, 7882, 7732, 5798, 2711, 2562, 5939, 2214, 2881, 2588, 10123, 6527, 10309, 2038, 2683, 5617, 2146, 4117, 10821, 2538, 5408, 5527, 6079, 7499, 6641, 2930, 5683, 6353, 5958, 2154, 5835] + [10015, 6339, 3168, 7481, 7779, 646, 4117, 7891, 4324, 2146, 10007, 1162, 10330, 285, 5527, 7616, 5617, 4196, 7771, 2085, 1104, 5512, 8303, 4409, 7343, 2538, 7570, 3977, 2683, 2038, 10250, 2494, 10309, 8423, 2417, 6353, 2564, 685, 6471, 6527, 10123, 2173, 2588, 205, 2881, 5863, 5958, 5408, 2607, 2214, 5939, 2562, 6079, 8208, 2154, 2799, 3909, 5798, 7663, 2574, 7732, 5835, 7882, 5766, 7556, 6438, 6641, 6359, 2761, 6210, 5434, 10232, 10043, 6109, 2323, 9550, 2437, 5456, 2930, 5683, 413, 2553, 7499, 6151, 5583, 10235, 6055, 7688, 5600, 5560, 3146, 5485, 10069, 5423, 5499, 6224, 5937, 292]
    }
    
    return labels

def get_images(cell_name, auxilin_dir=auxilin_dir):
    '''Loads in X and Y for one cell
    
    Returns
    -------
    X : np.ndarray
        has shape (W, H, num_images)
    Y : np.ndarray
        has shape (W, H, num_images)
    '''
    #cell_name = 'Cell1_1s'
    data_dir = oj(auxilin_dir, 'A7D2', cell_name) # 'A7D2', 'EGFB-GAK-F6'
    for name in os.listdir(oj(data_dir, 'TagRFP')):
        if 'tif' in name:
            fname1 = name
    for name in os.listdir(oj(data_dir, 'EGFP')):
        if 'tif' in name:
            fname2 = name
    X = imread(oj(data_dir, 'TagRFP', fname1)) #.astype(np.float32) # X = RFP(clathrin) (num_images x H x W)
    Y = imread(oj(data_dir, 'EGFP', fname2)) #.astype(np.float32) # Y = EGFP (auxilin) (num_image x H x W)  
    return X, Y

def get_tracks(cell_nums=[1, 2, 3, 4, 5, 6, 7, 8], all_data=False, processed_tracks_file='processed/tracks.pkl'):
    #if os.path.exists(processed_tracks_file):
    #    return pd.read_pickle(processed_tracks_file)
    dfs = []
    # 8 cell folders [1, 2, 3, ..., 8]
    for cell_num in cell_nums:
        fname = f'{auxilin_dir}/A7D2/Cell{cell_num}_1s/TagRFP/Tracking/ProcessedTracks.mat'
        cla, aux = get_images(f'Cell{cell_num}_1s', auxilin_dir=auxilin_dir)
        fname_image = data_dir = oj(auxilin_dir, 'A7D2', f'Cell{cell_num}_1s')
        mat = mat4py.loadmat(fname)
        tracks = mat['tracks']
        n = len(tracks['t'])
        totalDisplacement = []
        msd = [] # mean squared displacement
        for i in range(n):
            try:
                totalDisplacement.append(tracks['MotionAnalysis'][i]['totalDisplacement'])
            except:
                totalDisplacement.append(0)
            try:
                msd.append(np.nanmax(tracks['MotionAnalysis'][i]['MSD']))
            except:
                msd.append(0)

        CLATH = 0
        AUX = 1
                
        X = np.array([tracks['A'][i][CLATH] for i in range(n)])
        Y = np.array([tracks['A'][i][AUX] for i in range(n)])
        t = np.array([tracks['t'][i] for i in range(n)])
        x_pos_seq = np.array([tracks['x'][i][CLATH] for i in range(n)]) # x-position for clathrin (auxilin is very similar)
        y_pos_seq = np.array([tracks['y'][i][CLATH] for i in range(n)]) # y-position for clathrin (auxilin is very similar)
        X_pvals = np.array([tracks['pval_Ar'][i][CLATH] for i in range(n)])
        Y_pvals = np.array([tracks['pval_Ar'][i][AUX] for i in range(n)])
        
        # buffers
        X_starts = []
        Y_starts = []
        for d in tracks['startBuffer']:
            if len(d) == 0:
                X_starts.append([])
                Y_starts.append([])
            else:
                X_starts.append(d['A'][CLATH])
                Y_starts.append(d['A'][AUX])
        X_ends = []
        Y_ends = []
        for d in tracks['endBuffer']:
            if len(d) == 0:
                X_ends.append([])
                Y_ends.append([])
            else:
                X_ends.append(d['A'][CLATH])
                Y_ends.append(d['A'][AUX])
        X_extended = [X_starts[i] + X[i] + X_ends[i] for i in range(n)]
                
        
        # image feats
        pixel = np.array([[cla[int(t[i][j]), int(y_pos_seq[i][j]), int(x_pos_seq[i][j])]
                              if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                             for i in range(n)])
        pixel_up = np.array([[cla[int(t[i][j]), min(int(y_pos_seq[i][j] + 1), cla.shape[1] - 1), int(x_pos_seq[i][j])]
                              if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                             for i in range(n)])
        pixel_down = np.array([[cla[int(t[i][j]), max(int(y_pos_seq[i][j] - 1), 0), int(x_pos_seq[i][j])]
                              if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                             for i in range(n)])
        pixel_left = np.array([[cla[int(t[i][j]), int(y_pos_seq[i][j]), max(int(x_pos_seq[i][j]-1), 0)]
                              if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                             for i in range(n)])
        pixel_right = np.array([[cla[int(t[i][j]), int(y_pos_seq[i][j]), min(int(x_pos_seq[i][j]+1), cla.shape[2]-1)]
                              if not math.isnan(t[i][j]) else 0 for j in range(len(tracks['t'][i]))]
                             for i in range(n)])                            
        
        data = {
            'X': X,
            'X_extended': X_extended,
            'Y': Y,
            'X_starts': X_starts,
            'Y_starts': Y_starts,
            'X_ends': X_ends,
            'Y_ends': Y_ends,
            'X_pvals': X_pvals,
            'Y_pvals': Y_pvals,
            'catIdx': tracks['catIdx'],
            'total_displacement': totalDisplacement,
            'mean_square_displacement': msd,
            'lifetime': tracks['lifetime_s'],
            'lifetime_extended': [len(x) for x in X_extended],            
            'x_pos': [sum(x) / len(x) for x in x_pos_seq], # mean position in the image
            'y_pos': [sum(y) / len(y) for y in y_pos_seq],
            'cell_num': [cell_num] * n,
            't': [t[i][0] for i in range(n)],
            'x_pos_seq': x_pos_seq,
            'y_pos_seq': y_pos_seq,
        }
        if all_data:
            data['x_pos_seq'] = x_pos_seq
            data['y_pos_seq'] = y_pos_seq
            '''
            data['t'] = [t[i][0] for i in range(n)]
            data['pixel'] = pixel
            data['pixel_left'] = pixel_left
            data['pixel_right'] = pixel_right
            data['pixel_up'] = pixel_up
            data['pixel_down'] = pixel_down
            data['center_max'] = [max(pixel[i]) for i in range(n)],
            data['left_max'] = [max(pixel_left[i]) for i in range(n)],
            data['right_max'] = [max(pixel_right[i]) for i in range(n)],
            data['up_max'] = [max(pixel_up[i]) for i in range(n)],
            data['down_max'] = [max(pixel_down[i]) for i in range(n)],
            '''
        df = pd.DataFrame.from_dict(data)
        dfs.append(deepcopy(df))
    df = pd.concat(dfs)
    os.makedirs(os.path.dirname(processed_tracks_file), exist_ok=True)
    df.to_pickle(processed_tracks_file)
    return df

def preprocess(df):
    '''Add a bunch of extra features to the df
    '''
    df = df[df.lifetime > 2]
    df['X_max'] = np.array([max(x) for x in df.X.values])
    df['X_max_extended'] = np.array([max(x) for x in df.X_extended.values])
    df['X_min'] = np.array([min(x) for x in df.X.values])
    df['X_mean'] = np.nan_to_num(np.array([np.nanmean(x) for x in df.X.values]))
    df['X_std'] = np.nan_to_num(np.array([np.std(x) for x in df.X.values]))
    df['Y_max'] = np.array([max(y) for y in df.Y.values])    
    df['Y_mean'] = np.nan_to_num(np.array([np.nanmean(y) for y in df.Y.values]))
    df['Y_std'] = np.nan_to_num(np.array([np.std(y) for y in df.Y.values]))
    df['X_peak_idx'] = np.nan_to_num(np.array([np.argmax(x) for x in df.X]))
    df['Y_peak_idx'] = np.nan_to_num(np.array([np.argmax(y) for y in df.Y]))
    df['X_peak_time_frac'] = df['X_peak_idx'].values / df['lifetime'].values
    df['slope_end'] = df.apply(lambda row: (row['X_max'] - row['X'][-1]) / (row['lifetime'] - row['X_peak_idx']), axis=1)
    df['X_peak_last_15'] = df['X_peak_time_frac'] >= 0.85
    df['X_peak_last_5'] = df['X_peak_time_frac'] >= 0.95
    
    # hand-engineeredd features
    def calc_rise(x):
        '''max change before peak
        '''
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        return val_max - np.min(x[:idx_max + 1])

    def calc_fall(x):
        '''max change after peak
        '''
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        return val_max - np.min(x[idx_max:])
    
    def calc_rise_slope(x):
        '''slope to max change before peak
        '''
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        x_early = x[:idx_max + 1]
        idx_min = np.argmin(x_early)
        denom = (idx_max - idx_min)
        if denom == 0:
            return 0
        return (val_max - np.min(x_early)) / denom


    def calc_fall_slope(x):
        '''slope to max change after peak
        '''
        idx_max = np.argmax(x)
        val_max = x[idx_max]
        x_late = x[idx_max :]
        idx_min = np.argmin(x_late)
        denom = idx_min
        if denom == 0:
            return 0
        return (val_max - np.min(x_late)) / denom
    
    
    def max_diff(x): return np.max(np.diff(x))
    def min_diff(x): return np.min(np.diff(x))
    
    df['rise'] = df.apply(lambda row: calc_rise(row['X']), axis=1)
    df['fall'] = df.apply(lambda row: calc_fall(row['X']), axis=1)
    df['rise_extended'] = df.apply(lambda row: calc_rise(row['X_extended']), axis=1)
    df['fall_extended'] = df.apply(lambda row: calc_fall(row['X_extended']), axis=1)
    df['fall_late_extended'] = df.apply(lambda row: row['fall_extended'] if row['X_peak_last_15'] else row['fall'], axis=1)
    df['fall_final'] = df.apply(lambda row: row['X'][-3] - row['X'][-1], axis=1)
    
    df['rise_slope'] = df.apply(lambda row: calc_rise_slope(row['X']), axis=1)
    df['fall_slope'] = df.apply(lambda row: calc_fall_slope(row['X']), axis=1)
    num = 3
    df['rise_local_3'] = df.apply(lambda row: 
                                  calc_rise(np.array(row['X'][max(0, row['X_peak_idx'] - num): 
                                                            row['X_peak_idx'] + num + 1])), 
                                          axis=1)
    df['fall_local_3'] = df.apply(lambda row: 
                                  calc_fall(np.array(row['X'][max(0, row['X_peak_idx'] - num): 
                                                            row['X_peak_idx'] + num + 1])), 
                                          axis=1)
    
    num2 = 11
    df['rise_local_11'] = df.apply(lambda row: 
                                   calc_rise(np.array(row['X'][max(0, row['X_peak_idx'] - num2): 
                                                            row['X_peak_idx'] + num2 + 1])), 
                                          axis=1)
    df['fall_local_11'] = df.apply(lambda row: 
                                   calc_fall(np.array(row['X'][max(0, row['X_peak_idx'] - num2): 
                                                            row['X_peak_idx'] + num2 + 1])), 
                                          axis=1)
    df['max_diff'] = df.apply(lambda row: max_diff(row['X']), axis=1)    
    df['min_diff'] = df.apply(lambda row: min_diff(row['X']), axis=1)      
    
    
    # imputed feats
    d = df[['X_max', 'X_mean', 'lifetime', 'rise', 'fall']]
    d = d[df['X_peak_time_frac'] <= 0.8]
    m = RidgeCV().fit(d[['X_max', 'X_mean', 'lifetime', 'rise']], d['fall'])
    fall_pred = m.predict(df[['X_max', 'X_mean', 'lifetime', 'rise']])
    fall_imp = df['fall']
    fall_imp[df['X_peak_time_frac'] > 0.8] = fall_pred[df['X_peak_time_frac'] > 0.8]
    df['fall_imp'] = fall_imp
    
    return df


def add_rule_based_label(df):
    
    df['Y_peak_time_frac'] = df['Y_peak_idx'].values / df['lifetime'].values
    df['y_z_score'] = (df['Y_max'].values - df['Y_mean'].values)/df['Y_std'].values
    X_max_around_Y_peak = []
    X_max_after_Y_peak = []
    for i in range(len(df)):
        pt = df['Y_peak_idx'].values[i]
        lt = df['lifetime'].values[i]
        left_bf = np.int(0.2 * lt) + 1  # look at a window with length = 30%*lifetime
        right_bf = np.int(0.1 * lt) + 1
        arr_around = df['X'].iloc[i][max(0, pt - left_bf): min(pt + right_bf, lt)]
        arr_after = df['X'].iloc[i][min(pt + right_bf, lt - 1): ]
        X_max_around_Y_peak.append(max(arr_around))
        X_max_after_Y_peak.append(max(arr_after))
    df['X_max_around_Y_peak'] = X_max_around_Y_peak
    df['X_max_after_Y_peak'] = X_max_after_Y_peak
    df['X_max_diff'] = df['X_max_around_Y_peak'] - df['X_max_after_Y_peak']
    
    def rule_based_model(track):
    
        #three rules:
        #  if aux peaks too early -- negative
        #  elif:
        #     if y_consec_sig or y_conservative_thresh or (cla drops around aux peak, and aux max is greater than 
        #     mean + 2.6*std), then positive
        #  else: negative

        if track['Y_peak_time_frac'] < 0.2:
            return 0
        if track['y_consec_sig'] or track['y_conservative_thresh']:
            return 1
        #if track['X_max_diff'] > 260 and track['y_z_score'] > 2.6:
        #    return 1
        if track['X_max_diff'] > 260 and track['Y_max'] > 560:
            return 1
        return 0

    
    df['y_rule_based'] = np.array([rule_based_model(df.iloc[i]) for i in range(len(df))])
    return df


def add_outcomes(df, thresh=3.25, p_thresh=0.05, aux_peak=642.375, aux_thresh=973):
    '''Add binary outcome of whether spike happened and info on whether events were questionable
    '''
    df['y_score'] = df['Y_max'].values - (df['Y_mean'].values + thresh * df['Y_std'].values)
    df['y_thresh'] = (df['y_score'].values > 0).astype(np.int) # Y_max was big
    df['y'] = df['Y_max'] > aux_peak
    
    # outcomes based on significant p-values
    num_sigs = [np.array(df['Y_pvals'].iloc[i]) < p_thresh for i in range(df.shape[0])]
    df['y_num_sig'] = np.array([num_sigs[i].sum() for i in range(df.shape[0])]).astype(np.int)
    df['y_single_sig'] = np.array([num_sigs[i].sum() > 0 for i in range(df.shape[0])]).astype(np.int)
    df['y_double_sig'] = np.array([num_sigs[i].sum() > 1 for i in range(df.shape[0])]).astype(np.int)
    df['y_conservative_thresh'] = (df['Y_max'].values > aux_thresh).astype(np.int)
    y_consec_sig = []
    y_sig_min_diff = []
    for i in range(df.shape[0]):
        idxs_sig = np.where(num_sigs[i]==1)[0] # indices of significance
        if len(idxs_sig) > 1:
            y_sig_min_diff.append(np.min(np.diff(idxs_sig)))
        else:
            y_sig_min_diff.append(np.nan)
        # find whether there were consecutive sig. indices
        if len(idxs_sig) > 1 and np.min(np.diff(idxs_sig)) == 1:
            y_consec_sig.append(1)
        else:
            y_consec_sig.append(0)
    df['y_consec_sig'] = y_consec_sig
    df['y_sig_min_diff'] = y_sig_min_diff
    df['y_consec_thresh'] = np.logical_or(df['y_consec_sig'], df['y_conservative_thresh'])
    
    def add_hotspots(df, num_sigs, outcome_def='consec_sig'):
        '''Identify hotspots as any track which over its time course has multiple events
        events must meet the event definition, then for a time not meet it, then meet it again
        Example: two consecutive significant p-values, then non-significant p-value, then 2 more consecutive p-values
        '''

        if outcome_def=='consec_sig':
            hotspots = np.zeros(df.shape[0]).astype(np.int)
            for i in range(df.shape[0]):
                idxs_sig = np.where(num_sigs[i]==1)[0] # indices of significance
                if idxs_sig.size < 5:
                    hotspots[i] = 0
                else:
                    diffs = np.diff(idxs_sig)
                    consecs = np.where(diffs==1)[0] # diffs==1 means there were consecutive sigs
                    consec_diffs = np.diff(consecs)
                    if consec_diffs.shape[0] > 0 and np.max(consec_diffs) > 2: # there were greated than 2 non-consec sigs between the consec sigs
                        hotspots[i] = 1
                    else:
                        hotspots[i] = 0
        df['sig_idxs'] = num_sigs
        df['hotspots'] = hotspots
        
        return df
    
    df = add_hotspots(df, num_sigs)
    
    df['y_consec_thresh'][df.pid.isin(get_labels()['pos'])] = 1 # add manual pos labels
    df['y_consec_thresh'][df.pid.isin(get_labels()['neg'])] = 0 # add manual neg labels    
    df['hotspots'][df.pid.isin(get_labels()['hotspots'])] = 1 # add manual hotspot labels
    
    df = add_rule_based_label(df)
    
    return df

def remove_invalid_tracks(df, keep=[1, 2]):
    '''Remove certain types of tracks based on cat_idx
    cat_idx (idx 1 and 2)
    1-4 (non-complex trajectory - no merges and splits)
        1 - valid
        2 - signal occasionally drops out
        3 - cut  - starts / ends
        4 - multiple - at the same place (continues throughout)
    5-8 (there is merging or splitting)
    '''
    return df[df.catIdx.isin(keep)]


def extract_X_mat(df):
    '''Extract matrix for X filled with zeros after sequences
    Width of matrix is length of longest lifetime
    '''
    p = df.lifetime.max()
    n = df.shape[0]
    X_mat = np.zeros((n, p)).astype(np.float32)
    X = df['X'].values
    for i in range(n):
        x = X[i]
        num_timepoints = min(p, len(x))
        X_mat[i, :num_timepoints] = x[:num_timepoints]
    X_mat = np.nan_to_num(X_mat)
    X_mat -= np.min(X_mat)
    X_mat /= np.std(X_mat)
    return X_mat

def remove_tracks_by_lifetime(df: pd.DataFrame, outcome_def: str, plot=False, acc_thresh=0.95):
    '''Calculate accuracy you can get by just predicting max class 
    as a func of lifetime and return points within proper lifetime (only looks at training cells)
    '''
    vals = df[['lifetime', outcome_def]][df.cell_num.isin(cell_nums_train)]
    R, C = 1, 3
    lifetimes = np.unique(vals['lifetime'])

    # cumulative accuracy for different thresholds
    accs_cum_lower = np.array([1 - np.mean(vals[outcome_def][vals['lifetime'] <= l]) for l in lifetimes])
    idx_thresh = np.nonzero(accs_cum_lower >= acc_thresh)[0][-1] # last nonzero index
    thresh_lower = lifetimes[idx_thresh]    
    
    accs_cum_higher = np.array([np.mean(vals[outcome_def][vals['lifetime'] >= l]) for l in lifetimes]).flatten()
    try:
        idx_thresh_2 = np.nonzero(accs_cum_higher >= acc_thresh)[0][0]
        thresh_higher = lifetimes[idx_thresh_2]
    except:
        idx_thresh_2 = lifetimes.size - 1
        thresh_higher = lifetimes[idx_thresh_2] + 1
    
    n_short = np.sum(vals["lifetime"]<=thresh_lower)
    n_long = np.sum(vals["lifetime"]>=thresh_higher)
    acc_short = accs_cum_lower[idx_thresh]
    acc_long = accs_cum_higher[idx_thresh_2]

    if plot:
        plt.figure(figsize=(12, 4), dpi=200)
        plt.subplot(R, C, 1)
        outcome = df[outcome_def]
        plt.hist(df['lifetime'][outcome==1], label='aux+', alpha=1, color=cb, bins=25)
        plt.hist(df['lifetime'][outcome==0], label='aux-', alpha=0.7, color=cr, bins=25)
        plt.xlabel('lifetime')
        plt.ylabel('count')
        plt.legend()
    
        plt.subplot(R, C, 2)
        plt.plot(lifetimes, accs_cum_lower, color=cr)
    #     plt.axvline(thresh_lower)
        plt.axvspan(0, thresh_lower, alpha=0.2, color=cr)
        plt.ylabel('fraction of negative events')
        plt.xlabel(f'lifetime <= value\nshaded includes {n_short/n*100:0.0f}% of pts')


        plt.subplot(R, C, 3)
        plt.plot(lifetimes, accs_cum_higher, cb)
        plt.axvspan(thresh_higher, max(lifetimes), alpha=0.2, color=cb)
        plt.ylabel('fraction of positive events')
        plt.xlabel(f'lifetime >= value\nshaded includes {n_long/n*100:0.0f}% of pts')
        plt.tight_layout()
        plt.show()

    # only df with lifetimes in proper range
    idxs_valid = (df['lifetime'] > thresh_lower) & (df['lifetime'] < thresh_higher)
    df[~idxs_valid]['valid'] = 0
    metadata = {'num_short': n_short, 'num_long': n_long, 'acc_short': acc_short, 
                'acc_long': acc_long, 'thresh_short': thresh_lower, 'thresh_long': thresh_higher,
                'num_tracks_after_lifetime': df['valid'].sum(), 'num_aux_pos_after_lifetime': df[outcome_def].sum(),
                'num_hotspots_after_lifetime': df['hotspots'].sum()
               }
    return df, metadata

def add_dict_features(df, sc_comps_file='processed/dictionaries/sc_12_alpha=1.pkl', 
                      nmf_comps_file='processed/dictionaries/nmf_12.pkl', 
                      use_processed=True):
    
    '''Add features from saved dictionary to df
    '''
    def sparse_code(X_mat, n_comps=12, alpha=1, out_dir='processed/dictionaries'):
        print('sparse coding...')
        d = DictionaryLearning(n_components=n_comps, alpha=alpha, random_state=42)
        d.fit(X_mat)
        pkl.dump(d, open(oj(out_dir, f'sc_{n_comps}_alpha={alpha}.pkl'), 'wb'))

    def nmf(X_mat, n_comps=12, out_dir='processed/dictionaries'):
        print('running nmf...')
        d = NMF(n_components=n_comps, random_state=42)
        d.fit(X_mat)
        pkl.dump(d, open(oj(out_dir, f'nmf_{n_comps}.pkl'), 'wb'))
    
    X_mat = extract_X_mat(df)
    X_mat -= np.min(X_mat)
    
    # if feats don't exist, compute them
    if not use_processed or not os.path.exists(sc_comps_file):
        os.makedirs('processed/dictionaries', exist_ok=True)
        sparse_code(X_mat)
        nmf(X_mat)
    
    try:
        # sc
        d_sc = pkl.load(open(sc_comps_file, 'rb'))
        encoding = d_sc.transform(X_mat)
        for i in range(encoding.shape[1]):
            df[f'sc_{i}'] = encoding[:, i]

        # nmf
        d_nmf = pkl.load(open(nmf_comps_file, 'rb'))
        encoding_nmf = d_nmf.transform(X_mat)
        for i in range(encoding_nmf.shape[1]):
            df[f'nmf_{i}'] = encoding_nmf[:, i]
    except:
        print('dict features not added!')
    return df
                   
def add_smoothed_tracks(df, 
                        method='spline', 
                        s_spl=0.004):
    X_smooth_spl = []
    X_smooth_spl_dx = []
    X_smooth_spl_d2x = []
    def num_local_maxima(x):
        return(len([i for i in range(1, len(x)-1) if x[i] > x[i-1] and x[i] > x[i+1]]))
    for x in df['X']:
        spl = UnivariateSpline(x=range(len(x)), 
                               y=x, 
                               w=[1.0/len(x)]*len(x),
                               s=np.var(x)*s_spl)                   
        spl_dx = spl.derivative()
        spl_d2x = spl_dx.derivative()
        X_smooth_spl.append(spl(range(len(x))))
        X_smooth_spl_dx.append(spl_dx(range(len(x))))
        X_smooth_spl_d2x.append(spl_d2x(range(len(x))))
    df['X_smooth_spl'] = np.array(X_smooth_spl)
    df['X_smooth_spl_dx'] = np.array(X_smooth_spl_dx)
    df['X_smooth_spl_d2x'] = np.array(X_smooth_spl_d2x)
    df['X_max_spl'] = np.array([np.max(x) for x in X_smooth_spl])
    df['dx_max_spl'] = np.array([np.max(x) for x in X_smooth_spl_dx])
    df['d2x_max_spl'] = np.array([np.max(x) for x in X_smooth_spl_d2x])               
    df['num_local_max_spl'] = np.array([num_local_maxima(x) for x in X_smooth_spl])
    df['num_local_min_spl'] = np.array([num_local_maxima(-1 * x) for x in X_smooth_spl])
    
    # linear fits
    x = np.arange(5).reshape(-1, 1)
    df['end_linear_fit'] = [LinearRegression().fit(x, end).coef_[0] for end in df['X_ends']]
    df['start_linear_fit'] = [LinearRegression().fit(x, start).coef_[0] for start in df['X_starts']]
    return df
                   
    
def get_feature_names(df):
    '''Returns features (all of which are scalar)
    Removes metadata + time-series columns + outcomes
    '''
    ks = list(df.keys())
    feat_names = [
        k for k in ks
        if not k.startswith('y')
        and not k.startswith('Y')
        and not k.startswith('pixel')
#         and not k.startswith('pc_')
        and not k in ['catIdx', 'cell_num', 'pid', 'valid', # metadata
                      'X', 'X_pvals', 'x_pos', 'X_starts', 'X_ends', 'X_extended',
                      'X_peak_idx',
                      'hotspots', 'sig_idxs',
                      'X_max_around_Y_peak',
                      'X_max_after_Y_peak',                      
                      'X_max_diff',
                      't','x_pos_seq','y_pos_seq',
                      'X_smooth_spl', 'X_smooth_spl_dx', 'X_smooth_spl_d2x'] # curves not features
    ]
    return feat_names


def add_pcs(df):
    '''adds 10 pcs based on feature names
    '''
    feat_names = get_feature_names(df)
    X = df[feat_names]
    X = (X - X.mean()) / X.std()
    pca = decomposition.PCA(whiten=True)
    pca.fit(X[~df.cell_num.isin(cell_nums_test)])
    X_reduced = pca.transform(X)  
    for i in range(10):
        df['pc_' + str(i)] = X_reduced[:, i]
    return df