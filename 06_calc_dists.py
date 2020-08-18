
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
outcome_def = 'y_consec_thresh'

df = data.get_data()
df = df[df['valid'] == 1] # exclude test cells, short/long tracks, hotspots
viz.print_metadata()

def sort_outcome(df, outcome_def):
    outcome_score = df['Y_max'].values
    idxs_sort = np.argsort(outcome_score)
    return df[outcome_def].values[idxs_sort], df['Y'].values[idxs_sort]

outcome_sort, Y_sort = sort_outcome(df, outcome_def)

from scipy.spatial.distance import euclidean

from fastdtw import fastdtw


n = df.shape[0]
dists = np.zeros((n, n))
for i in tqdm(range(n)):
    for j in range(n - i):
        x0 = np.array(df['X'].iloc[i])
        x1 = np.array(df['X'].iloc[j])
        distance, path = fastdtw(x0, x1, dist=euclidean)
        dists[i, j] = distance
    with open('dists_dtw.npy', 'wb') as f:
        np.save(f, dists)