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
from scipy import ndimage
import math
import mat4py
import collections
import tracks_common

def find_clathrin_neighbors(tracks, tracks_by_frame, distance, norm_ord):
    
    """
    function that populates the "neighbors" attribute of all clathrin signals
    
    Input: 
        tracks, tracks_by_frame: output of read_tracks
        distance: float
        norm_ord: np.inf or int
            this will be used for the ord argument of np.linalg.norm
            np.inf corresponds to square patches
            
    Output:
        list of tracks, for each clathrin signal in each track, this computes all neighbors of that signal within distance
        (in norm_ord norm) smaller than elements in the "distance" argument
    """
    
    distance = [dist for dist in distance if dist not in tracks[0].clathrin_signals[0].neighbors]
    for track in tracks:
        for signal in track.clathrin_signals:
            for dist in distance:
                signal.neighbors[dist] = []
            for ss in tracks_by_frame[signal.frame]:
                ds = np.linalg.norm(np.array((signal.x - ss['cla'].x, signal.y - ss['cla'].y)), 
                                    ord=norm_ord)
                for dist in distance:
                    if ds <= dist and ds > 0.01:
                        signal.neighbors[dist].append(ss['cla'])
    return tracks


#def num_of_neighbors(tracks, tracks_by_frame, distance, norm_ord):
    #to be finished...
        
    
    
    
    