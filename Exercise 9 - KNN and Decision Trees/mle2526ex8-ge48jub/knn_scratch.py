import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

def minkowski_distance(a, b, p=2):
    """
    Compute Minkowski distance between two vectors.
    D(a, b) = (sum(|a_i - b_i|^p))^(1/p)
    """
    # 1. Compute absolute difference |a - b|
    diff = np.abs(a - b)
    # 2. Raise to power p, sum, and take 1/p root
    distance = np.sum(diff ** p) ** (1.0 / p)
    return distance

def compute_weights(distances):
    """
    Compute inverse distance weights.
    Weights w_i = 1 / (d_i + epsilon).
    Normalize so they sum to 1.
    """
    epsilon = 1e-8
    weights = 1.0 / (distances + epsilon)
    weights = weights / np.sum(weights)
    return weights

class CustomKNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, k=5, p=2, weight_mode='uniform'):
        """
        k: Number of neighbors
        p: Power for Minkowski distance (1=Manhattan, 2=Euclidean)
        weight_mode: 'uniform' or 'distance'
        """
        self.k = k
        self.p = p
        self.weight_mode = weight_mode
        self.X_train = None
        self.y_train = None
        self.neighbor_finder = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        Fit the model using sklearn's NearestNeighbors for indexing.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

        # update self.X_train by applying scaler fit_transform
        self.X_train = self.scaler.fit_transform(self.X_train)

        metric = lambda a, b: minkowski_distance(a, b, self.p)

        # Initialize NearestNeighbors with n_neighbors=self.k and metric=metric
        # Then fit it on self.X_train
        self.neighbor_finder = NearestNeighbors(n_neighbors=self.k, metric=metric)
        self.neighbor_finder.fit(self.X_train)

        return self

    def predict(self, X):
        """
        Predict target values. 
        Note: We manually calculate distances to ensure we use our 'minkowski_distance' function
        logic and to apply custom weighting.
        """
        X = np.array(X)
        # 1. Apply scaler so that X is in the same space as X_train
        X = self.scaler.transform(X)

        # 2. Get distances and indices of k nearest neighbors
        # you should use the function kneighbors from self.neighbor_finder
        distances, indices = self.neighbor_finder.kneighbors(X)
        
        predictions = []
        for i, idx_row in enumerate(indices):
            # idx_row contains indices of neighbors for the i-th query point
            neighbors_y = self.y_train[idx_row]
            
            # 3. Compute weights
            if self.weight_mode == 'distance':
                # 3. Use compute_weights on distances to get weights
                # Recall that distances[i, :] contains distances for the i-th query point
                # you should only consider the first self.k distances
                # You can use numpy array slicing for that, by defining ':value' as the upper limit
                weights = compute_weights(distances[i, :self.k])
            else:
                weights = np.ones(self.k) / self.k
            
            # 4. Weighted average
            pred = np.sum(weights * neighbors_y)
            predictions.append(pred)
            
        return np.array(predictions)