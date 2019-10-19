import numpy as np
from torch import nn
import torch.nn.functional as F
import torch


class MaxLinear(nn.Module):
    '''Takes flattened input and predicts it using many linear units
    '''
    def __init__(self, input_dim=24300, num_units=20, nonlin=F.relu, use_bias=False):
        super(MaxLinear, self).__init__()

        self.fc1 = nn.Linear(input_dim, num_units, bias=use_bias)

    def forward(self, X, **kwargs):
#         print('in shape', X.shape, X.dtype)
        X = self.fc1(X) #.max(dim=-1)
#         print('out shape', X.shape, X.dtype)
        X = torch.max(X, dim=1)[0] # 0 because this returns max, indices
#         print('out2 shape', X.shape, X.dtype)
        return X


class MaxConv(nn.Module):
    '''Takes flattened input and predicts it using many conv unit
    '''
    def __init__(self, input_dim=24300, num_units=20, kernel_size=30, nonlin=F.relu, use_bias=False):
        super(MaxConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_units, kernel_size=kernel_size, bias=use_bias)
#         torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

    def forward(self, X, **kwargs):
#         print('in shape', X.shape, X.dtype)
        X = self.conv1(X) #.max(dim=-1)
#         print('out shape', X.shape, X.dtype)
        # max over channels
        X = torch.max(X, dim=1)[0] # 0 because this returns max, indices
        
        # max over time step
        X = torch.max(X, dim=1)[0] # 0 because this returns max, indices
#         print('out2 shape', X.shape, X.dtype)
        
        X = X.unsqueeze(1)
        return X