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
#         self.offset = nn.Parameter(torch.Tensor([0]))

    def forward(self, X, **kwargs):
#         print('in shape', X.shape, X.dtype)
        X = self.fc1(X) #.max(dim=-1)
#         print('out shape', X.shape, X.dtype)
        X = torch.max(X, dim=1)[0] # 0 because this returns max, indices
#         print('out2 shape', X.shape, X.dtype)
        return X # + self.offset


class MaxConv(nn.Module):
    '''Takes flattened input and predicts it using many conv unit
        X: batch_size x 1 x num_timepoints
    '''
    def __init__(self, input_dim=24300, num_units=20, kernel_size=30, nonlin=F.relu, use_bias=False):
        super(MaxConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_units, kernel_size=kernel_size, bias=use_bias)
#         torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.offset = nn.Parameter(torch.Tensor([0]))

    def forward(self, X, **kwargs):
#         print('in shape', X.shape, X.dtype)
        X = X.unsqueeze(1)
        X = self.conv1(X) #.max(dim=-1)
#         print('out shape', X.shape, X.dtype)
        # max over channels
        X = torch.max(X, dim=1)[0] # 0 because this returns max, indices
        
        # max over time step
        X = torch.max(X, dim=1)[0] + self.offset # 0 because this returns max, indices
#         print('out2 shape', X.shape, X.dtype)
        
        X = X.unsqueeze(1)
        return X
    
class MaxConvLinear(nn.Module):
    '''Takes input patch, uses linear filter to convert it to time series, then runs temporal conv, then takes max
        X: batch_size x H_patch x W_patch x time
    '''
    def __init__(self, num_timepoints=300, num_linear_filts=1, num_conv_filts=3, patch_size=9, 
                 kernel_size=30, nonlin=F.relu, use_bias=False):
        super(MaxConvLinear, self).__init__()
        self.fc1 = nn.Linear(patch_size * patch_size, num_linear_filts, bias=use_bias) 
        self.conv1 = nn.Conv1d(in_channels=num_linear_filts, out_channels=num_conv_filts, kernel_size=kernel_size, bias=use_bias)
        self.offset = nn.Parameter(torch.Tensor([0]))

    def forward(self, X, **kwargs):
        s = X.shape # batch_size x H_patch x W_patch x time
        X = X.reshape(s[0], s[1] * s[2], s[3])
        X = torch.transpose(X, 1, 2)
#         print('in shape', X.shape, X.dtype)
        X = self.fc1(X) #.max(dim=-1)
        X = torch.transpose(X, 1, 2)

        X = self.conv1(X) #.max(dim=-1)
#         print('out shape', X.shape, X.dtype)
        # max over channels
        X = torch.max(X, dim=1)[0] # 0 because this returns max, indices
        
        # max over time step
        X = torch.max(X, dim=1)[0] #+ self.offset # 0 because this returns max, indices
#         print('out2 shape', X.shape, X.dtype)
        
        X = X.unsqueeze(1)
        return X