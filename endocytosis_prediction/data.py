from matplotlib import pyplot as plt
import numpy as np
from skimage.external.tifffile import imread
from skimage import io
import os
from os.path import join as oj
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, train_test_split


def get_data(auxilin_dir = '/scratch/users/vision/data/abc_data/auxilin_data'):
    '''Loads in X and Y for one cell
    
    Returns
    -------
    X : np.ndarray
        has shape (W, H, num_images)
    Y : np.ndarray
        has shape (W, H, num_images)
    '''
    
    cell_name = 'Cell1_1s'
    data_dir = oj(auxilin_dir, 'A7D2', cell_name) # 'A7D2', 'EGFB-GAK-F6'
    fname1 = os.listdir(oj(data_dir, 'TagRFP'))[0]
    fname2 = os.listdir(oj(data_dir, 'EGFP'))[0]
    X = imread(oj(data_dir, 'TagRFP', fname1)) # X = RFP(clathrin) (num_images x H x W)
    Y = imread(oj(data_dir, 'EGFP', fname2)) # Y = EGFP (auxilin) (num_image x H x W)
    return X, Y

def extract_single_pixel_features(X, Y):
    '''Extract time-series for single pixels as features
    '''
    X_feat = X.transpose() # W x H x num_images
    X_feat = X_feat.reshape(X_feat.shape[0] * X_feat.shape[1], -1) # num_pixels x num_images
    y_max = np.expand_dims(Y.sum(axis=0).flatten(), 1) # num_pixels x 1
    return X_feat, y_max

def extract_patch_features(X, Y, patch_size=9):
    '''Extract time-series for patches as features
    
    Returns
    -------
    X : np.ndarray
        has shape (num_patches, patch_size, patch_size, num_images)
    Y : np.ndarray
        has shape (num_images)
    '''
    X_feat = X.transpose() # W x H x num_images
    X_patches = extract_patches_2d(X_feat, patch_size=(patch_size, patch_size), max_patches=None) # num_patches x patch_size x patch_size x num_images
#     X_patches_flat = X_patches.reshape(X_patches.shape[0], -1)
    
    
    # take only the center of the y matches
    Y_max = np.max(Y, axis=0) # H x W
    Y_patches = extract_patches_2d(Y_max.transpose(), patch_size=(patch_size, patch_size), max_patches=None) # num_patches x patch_size x patch_size x num_images
    patch_center = patch_size // 2
    Y_centers = Y_patches[:, patch_center, patch_center]
    
    return X_patches, Y_centers