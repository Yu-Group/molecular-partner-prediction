import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import torch.utils.data as data_utils
from features import downsample


class fcnn(nn.Module):
    
    """
    customized (one hidden layer) fully connected neural network class
    """

    def __init__(self, D_in, H, p):
        
        """
        Parameters:        
        ==========================================================
            D_in: int
                dimension of input track
                
            H: int
                hidden layer size
                
            p: int
                number of additional covariates (such as lifetime, msd, etc..., to be concatenated to the hidden layer)            
        """

        super(fcnn, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        #self.fc2 = nn.Linear(H, H)
        self.bn1 = nn.BatchNorm1d(H)
        self.fc2 = nn.Linear(H + p, 1) 
    
    def forward(self, x1, x2):
        
        z1 = self.fc1(x1)
        z1 = self.bn1(z1)
        h1 = F.relu(z1)
        if x2 is not None:
            h1 = torch.cat((h1, x2), 1)
        z2 = self.fc2(h1)
        #h2 = F.relu(z2)
        #z3 = self.fc3(h2)       
        
        return F.relu(z2)
    
class fcnn_sklearn():
    
    """
    sklearn wrapper for the customized fcnn class
    """
    
    def __init__(self, D_in, H, p, epochs, batch_size, track_name, torch_seed=2):
        
        """
        Parameters:
        ==========================================================
            D_in, H, p: int
                same as input to fcnn
                
            epochs: int
                number of epochs
                
            batch_size: int
                batch size
                
            track_name: str
                column name of track (the tracks should be of the same length)
        """
        
        torch.manual_seed(torch_seed)
        self.D_in = D_in
        self.H = H
        self.p = p
        self.epochs = epochs
        self.batch_size = batch_size
        self.track_name = track_name
        self.torch_seed = torch_seed
        self.model = fcnn(D_in, H, p)
        
    def fit(self, X, y):
        
        """
        Train model
        
        Parameters:
        ==========================================================
            X: pd.DataFrame
                input data, should contain tracks and additional covariates
                
            y: np.array
                input response
        """        
        
        torch.manual_seed(self.torch_seed)
        
        # initialize model
        self.model = fcnn(self.D_in, self.H, self.p)
        
        # convert input dataframe to tensors
        X_track = X[self.track_name] # track
        X_track = torch.tensor(np.array(list(X_track.values)), dtype=torch.float)
        
        if len(X.columns) > 1: # covariates
            X_covariates = X[[c for c in X.columns if c != self.track_name]]
            X_covariates = torch.tensor(np.array(X_covariates).astype(float), dtype=torch.float)
        else:
            X_covariates = None
            
        # response
        y = torch.tensor(y.reshape(-1, 1), dtype=torch.float)
        
        # initialize optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # initialize dataloader
        dataset = torch.utils.data.TensorDataset(X_track, X_covariates, y)
        train_loader = torch.utils.data.DataLoader(dataset, 
                                                   batch_size=self.batch_size,
                                                   shuffle=True) 
        #train_loader = [(X1, X2, y)]
        
        # train fcnn
        for epoch in range(self.epochs):
            train_loss = 0
            for batch_idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                preds = self.model(data[0], data[1])
                loss_fn = torch.nn.MSELoss()
                loss = loss_fn(preds, data[2])
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            if epoch % 100 == 99:
                print(f'Epoch: {epoch}, Average loss: {train_loss/len(X_track)}')
            
    def predict(self, X_new):
        
        """
        make predictions with new data
        
        Parameters:
        ==========================================================
            X_new: pd.DataFrame
                input new data, should contain tracks and additional covariates
        """ 
        
        # convert input dataframe to tensors
        X_new_track = X_new[self.track_name]
        X_new_covariates = X_new[[c for c in X_new.columns if c != self.track_name]]
        X_new_track = torch.tensor(np.array(list(X_new_track.values)), dtype=torch.float)
        X_new_covariates = torch.tensor(np.array(X_new_covariates).astype(float), dtype=torch.float)        
        #X_new = torch.tensor(np.array(list(X_new.values)), dtype=torch.float)
        
        # make predictions
        self.model.eval()
        with torch.no_grad():            
            preds = self.model(X_new_track, X_new_covariates)
            
        return preds.data.numpy().reshape(1, -1)[0]
        
        