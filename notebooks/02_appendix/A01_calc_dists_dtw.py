import sys
sys.path.append('..')
import config
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('dark_background')
import data
from util.style import *
from tqdm import tqdm
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw


outcome_def = 'y_consec_thresh'

df = data.get_data()
df = df[df['valid'] == 1] # exclude test cells, short/long tracks, hotspots
n = df.shape[0]
dists = np.zeros((n, n))
for i in tqdm(range(n)):
    for j in range(i + 1, n):
        x0 = np.array(df['X'].iloc[i])
        x1 = np.array(df['X'].iloc[j])
        distance, path = fastdtw(x0, x1, dist=euclidean)
        dists[i, j] = distance
    with open(oj(config.DIR_INTERIM, 'dists_dtw.npy'), 'wb') as f:
        np.save(f, dists)