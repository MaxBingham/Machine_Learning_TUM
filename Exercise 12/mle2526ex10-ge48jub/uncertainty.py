import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim

class Committee:
    """
    Task 3: Bagging Committee for Uncertainty.
    """
    def __init__(self, base_estimator, n_estimators=10):
        self.estimators = []
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.estimators = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            est = copy.deepcopy(self.base_estimator)
            est.fit(X_sample, y_sample)
            self.estimators.append(est)
        return self

    def predict_uncertainty(self, X):
        """
        Returns:
            mu: Mean prediction (N,)
            sigma: Standard deviation of predictions (N,)
        """

        #loop through every estimator and save its results to list
        predictions = [] 
        for est in self.estimators:
            pred = est.predict(X)
            predictions.append(pred)
        predictions = np.array(predictions)  # shape: (n_estimators, n_samples)
        mu = np.mean(predictions, axis=0)
        sigma = np.std(predictions, axis=0)
        return mu, sigma

class DropoutNet(nn.Module):
    """
    Task 4 & 5: MC Dropout Neural Network.
    """
    def __init__(self, input_dim=2, hidden_dim=64, dropout_rate=0.2):
        super(DropoutNet, self).__init__()
        """
        Task 4: Define the network architecture.
        It should have at least 2 hidden layers with ReLU activations.
        Crucially: Add nn.Dropout(p=dropout_rate) layers.
        """
        self.model = None 
       
    #2 hidden layers 
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), 
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate), 
            nn.Linear(hidden_dim, 1)


        )
        
        
        # Loss
        self.criterion = nn.MSELoss()
        
    def forward(self, x):
        if self.model is None: return x
        return self.model(x)

    def fit(self, X, y, epochs=200, lr=0.01):
        # Conversion to tensors
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(y).view(-1, 1)
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train() # Set to training mode
        
        for _ in range(epochs):
            optimizer.zero_grad()
            output = self(X_t)
            loss = self.criterion(output, y_t)
            loss.backward()
            optimizer.step()
        return self

    def predict_uncertainty(self, X, n_passes=20):
        """
        Task 5: Monte Carlo Dropout Sampling.
        1. Convert X to tensor.
        2. Set model to train mode (self.train()) to ensure dropout is active.
        3. Run n_passes forward passes.
        4. Compute mean and std of results (detach and convert to numpy).
        """
        tensor_X = torch.FloatTensor(X) 
        self.train() 
        predictions = []
        for _ in range(n_passes): 
            output = self(tensor_X)
            predictions.append(output.detach().cpu().numpy().flatten())
        predictions = np.array(predictions)

        mu = predictions.mean(axis=0)
        std = predictions.std(axis=0)


        

        return mu, std