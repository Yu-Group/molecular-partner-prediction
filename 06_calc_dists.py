
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import numpy as np
plt.style.use('dark_background')
import data
from sklearn.model_selection import KFold
from util.style import *
import itertools
from sklearn import mixture
from matplotlib_venn import venn3, venn2
import viz
import config
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
    with open('dists_dtw.npy', 'wb') as f:
        np.save(f, dists)