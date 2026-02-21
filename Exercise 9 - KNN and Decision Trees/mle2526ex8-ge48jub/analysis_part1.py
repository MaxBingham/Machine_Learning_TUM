import numpy as np
from sklearn.model_selection import GridSearchCV
from knn_scratch import CustomKNNRegressor

def simulate_curse(d_values=[2, 10, 100, 1000], n_samples=1000):
    """
    Simulate the Curse of Dimensionality.
    1. For each dimension d:
       a. Generate n_samples random points in [0,1]^d
       b. Compute pairwise distances from the *first* point to all others (or origin to all)
       c. Compute contrast ratio: (max_dist - min_dist) / min_dist
    
    Returns: dict {dimension: ratio}
    """
    ratios = {}
    # Use numpy for efficient generation. 
    # np.linalg.norm(X, axis=1) computes Euclidean distances.
    for d in d_values:
        # Generate n_samples random points in [0,1]^d
        X = np.random.rand(n_samples, d)
        
        # Compute pairwise distances from the first point to all others
        first_point = X[0]
        distances = np.linalg.norm(X - first_point, axis=1)
        
        # Remove the distance to itself (which is 0)
        distances = distances[1:]
        
        # Compute contrast ratio: (max_dist - min_dist) / min_dist
        max_dist = np.max(distances)
        min_dist = np.min(distances)
        ratio = (max_dist - min_dist) / min_dist
        
        ratios[d] = ratio
    
    return ratios

def tune_knn_hyperparameters(X_train, y_train):
    """
    Use GridSearchCV to find the best k and p for CustomKNNRegressor.
    
    Search space:
    - k: [1, 3, 5, 10, 20]
    - p: [1, 2] (Manhattan vs Euclidean)
    - weight_mode: ['distance'] (fixed)
    
    Returns: best_params_ (dict)
    """
    # 1. Initialize CustomKNNRegressor
    knn = CustomKNNRegressor()
    
    # 2. Define param_grid
    param_grid = {
        'k': [1, 3, 5, 10, 20],
        'p': [1, 2],
        'weight_mode': ['distance']
    }
    
    # 3. Setup GridSearchCV with cv=3, scoring='neg_mean_squared_error'
    grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='neg_mean_squared_error')
    
    # 4. Fit
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_