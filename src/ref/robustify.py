import numpy as np


def robust_measure(df, func):
    original_feature = np.array([func(df.iloc[i].X) for i in range(len(df))])
    res = []
    for noise_level in range(1, 500, 20):
        new_feature = np.array([func(df.iloc[i].X + 
                                     noise_level*np.random.normal(size=len(df.iloc[i].X))) 
                                for i in range(len(df))])
        res.append(np.corrcoef(np.transpose(original_feature), np.transpose(new_feature))[0,1])
    return res

