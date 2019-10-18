import numpy as np
from torch import nn
import torch.nn.functional as F
import torch


class MaxLinear(nn.Module):
    '''Takes flattened input and predicts it using many linear units
    '''
    def __init__(self, input_dim=24300, num_units=20, nonlin=F.relu):
        super(MaxLinear, self).__init__()

        self.fc1 = nn.Linear(input_dim, num_units)

    def forward(self, X, **kwargs):
#         print('in shape', X.shape, X.dtype)
        X = self.fc1(X) #.max(dim=-1)
#         print('out shape', X.shape, X.dtype)
        X = torch.max(X, dim=1)[0] # 0 because this returns max, indices
#         print('out2 shape', X.shape, X.dtype)
        return X