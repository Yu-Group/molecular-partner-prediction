import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

sys.path.append('lib')
import collections

cell_nums_feature_selection = np.array([1])
cell_nums_train = np.array([1, 2, 3, 4, 5])
cell_nums_test = np.array([6])


def get_rf_neighbors(df, feat_names, outcome_def='y_thresh',
                     balancing='ros', balancing_ratio=1, out_name='results/classify/test.pkl',
                     feature_selection=None, feature_selection_num=3, seed=42):
    # pre-processing same as train.train

    np.random.seed(seed)
    X = df[feat_names]
    y = df[outcome_def].values

    m = RandomForestClassifier(n_estimators=100)

    kf = KFold(n_splits=len(cell_nums_train))

    # feature selection on cell num 1    
    feature_selector = None
    if feature_selection is not None:
        if feature_selection == 'select_lasso':
            feature_selector_model = Lasso()
        elif feature_selection == 'select_rf':
            feature_selector_model = RandomForestClassifier()
        # select only feature_selection_num features
        feature_selector = SelectFromModel(feature_selector_model, threshold=-np.inf,
                                           max_features=feature_selection_num)
        idxs = df.cell_num.isin(cell_nums_feature_selection)
        feature_selector.fit(X[idxs], y[idxs])
        X = feature_selector.transform(X)
        support = np.array(feature_selector.get_support())
    else:
        support = np.ones(len(feat_names)).astype(np.bool)

    # split testing data based on cell num
    idxs_test = df.cell_num.isin(cell_nums_test)
    X_test, Y_test = X[idxs_test], y[idxs_test]
    idxs_train = df.cell_num.isin(cell_nums_train)
    X_train, Y_train = X[idxs_train], y[idxs_train]
    # num_pts_by_fold_cv = []

    # build dictionary, key is leaf node, value is list of training samples in the node

    m.fit(X_train, Y_train)
    node_indices = m.apply(X_train)
    node_indices_test = m.apply(X_test)
    similarity_matrix = np.zeros((len(X_test), len(X_train)))
    for tree in range(100):
        node_samples = collections.defaultdict(list)
        for i in range(len(X_train)):
            node_samples[node_indices[i, tree]].append(i)
        for i in range(len(X_test)):
            node = node_indices_test[i, tree]
            for j in node_samples[node]:
                similarity_matrix[i, j] += 1
    preds_proba = m.predict_proba(X_test)[:, 1]

    # nearest neighbors and similarity
    nearest_neighbors = [np.argsort(similarity_matrix[i, :])[::-1][:10] for i in range(len(X_test))]
    similarity = [np.sort(similarity_matrix[i, :])[::-1][:10] for i in range(len(X_test))]

    idxs_test = np.where(idxs_test == True)
    idxs_train = np.where(idxs_train == True)
    df_train = df.iloc[idxs_train]
    df_test = df.iloc[idxs_test]

    df_test['preds_proba'] = preds_proba
    df_test['nearest_neighbors'] = nearest_neighbors
    df_test['similarity'] = similarity

    return df_train, df_test
