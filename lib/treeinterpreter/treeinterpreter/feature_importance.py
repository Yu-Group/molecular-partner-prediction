from .treeinterpreter import _predict_tree
from sklearn.ensemble.forest import _generate_unsampled_indices, _generate_sample_indices
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
def MDA(rf, X, y, type = 'oob', n_trials = 10, metric = 'accuracy'):
    if len(y.shape) != 2:
        raise ValueError('y must be 2d array (n_samples, 1) if numerical or (n_samples, n_categories).')
    n_samples, n_features = X.shape
    fi_mean = np.zeros((n_features,))
    fi_std = np.zeros((n_features,))
    best_score = rf_accuracy(rf, X, y, type = type, metric = metric)
    for f in range(n_features):
        permute_score = 0
        permute_std = 0
        X_permute = X.copy()
        for i in range(n_trials):
            X_permute[:, f] = np.random.permutation(X_permute[:, f])
            to_add = rf_accuracy(rf, X_permute, y, type = type, metric = metric)
            permute_score += to_add
            permute_std += to_add ** 2
        permute_score /= n_trials
        permute_std /= n_trials
        permute_std = (permute_std - permute_score ** 2) ** .5 / n_trials ** .5
        fi_mean[f] = best_score - permute_score 
        fi_std[f] = permute_std
    return fi_mean, fi_std
         
def neg_mse(y, y_hat):
    return - mean_squared_error(y, y_hat)

def rf_accuracy(rf, X, y, type = 'oob', metric = 'accuracy'):
    if metric == 'accuracy':
        score = accuracy_score
    elif metric == 'mse':
        score = neg_mse
    else:
        raise ValueError('metric type not understood')

    n_samples, n_features = X.shape
    tmp = 0
    count = 0
    if type == 'test':
        return score(y, rf.predict(X))
    elif type == 'train' and not rf.bootstrap:
        return score(y, rf.predict(X))

    for tree in rf.estimators_:
        if type == 'oob':
            if rf.bootstrap:
                indices = _generate_unsampled_indices(tree.random_state, n_samples)
            else:
                raise ValueError('Without bootstrap, it is not possible to calculate oob.')
        elif type == 'train':
            indices = _generate_sample_indices(tree.random_state, n_samples)
        else:
            raise ValueError('type is not recognized. (%s)'%(type))
        tmp +=  score(y[indices,:], tree.predict(X[indices, :])) * len(indices) 
        count += len(indices)
    return tmp / count
    
    
def feature_importance(rf, X, y, type = 'oob', normalized = False, balanced = False):
    n_samples, n_features = X.shape
    if len(y.shape) != 2:
        raise ValueError('y must be 2d array (n_samples, 1) if numerical or (n_samples, n_categories).')
    out = np.zeros((n_features,))
    SE = np.zeros((n_features,))
    for tree in rf.estimators_:
        if type == 'oob':
            if rf.bootstrap:
                indices = _generate_unsampled_indices(tree.random_state, n_samples)
            else:
                raise ValueError('Without bootstrap, it is not possible to calculate oob.')
        elif type == 'test':
            indices = np.arange(n_samples)
        elif type == 'classic':
            if rf.bootstrap:
                indices = _generate_sample_indices(tree.random_state, n_samples)
            else:
                indices = np.arange(n_samples)
        else:
            raise ValueError('type is not recognized. (%s)'%(type))
        _, _, contributions = _predict_tree(tree, X[indices,:])
        if balanced and (type == 'oob' or type == 'test'):
            base_indices = _generate_sample_indices(tree.random_state, n_samples)
            ids = tree.apply(X[indices, :])
            base_ids = tree.apply(X[base_indices, :])
            tmp1, tmp2 = np.unique(ids, return_counts = True)
            weight1 = {key: 1. / value for key, value in zip(tmp1, tmp2)}
            tmp1, tmp2 = np.unique(base_ids, return_counts = True)
            weight2 = {key: value for key, value in zip(tmp1, tmp2)}
            final_weights = np.array([[weight1[id] * weight2[id]] for id in ids])
            final_weights /= np.mean(final_weights)
        else:
            final_weights = 1
        if len(contributions.shape) == 2:
            contributions = contributions[:,:,np.newaxis]
        #print(contributions.shape)
        #print(np.array(y[indices,:]).shape)
        tmp =  np.tensordot(np.array(y[indices,:]) * final_weights, contributions, axes=([0, 1], [0, 2])) 
        if normalized:
            out +=  tmp / sum(tmp)
        else:
            out += tmp / len(indices)
        if normalized:
            SE += (tmp / sum(tmp)) ** 2
        else:
            SE += (tmp / len(indices)) ** 2
    out /= rf.n_estimators
    SE /= rf.n_estimators
    SE = ((SE - out ** 2) / rf.n_estimators) ** .5 
    return out, SE
    #if np.sum(out[out > 0]) + 10 * np.sum(out[out < 0]) < 0:
    #    return out
    #else:
    #    return out / np.sum(out) 
