import numpy as np
from scipy.spatial.distance import cdist

def rbf_kernel(X1, X2, gamma=1.0):
    """
    Task 1: Radial Basis Function (Gaussian) Kernel.
    Computes exp(-gamma * ||x1 - x2||^2).
    
    Args:
        X1: Array of shape (N, D)
        X2: Array of shape (M, D)
        gamma: float, length scale parameter
    Returns:
        K: Array of shape (N, M)
    """
    sq_dists = cdist(X1, X2, metric='sqeuclidean')

    K = np.exp(-gamma*sq_dists)

    return K
   

class KernelRidgeRegressor:
    """
    Task 2: Kernel Ridge Regression (Dual Formulation).
    """
    def __init__(self, gamma=1.0, alpha_reg=1e-5):
        self.gamma = gamma
        self.alpha_reg = alpha_reg 
        self.X_fit_ = None
        self.dual_coef_ = None 

    def fit(self, X, y):
        """
        1. Compute Gram Matrix K using rbf_kernel.
        2. Solve (K + lambda*I) * a = y for a.
        3. Store X and a.
        
        Hint: Use np.eye() for identity and np.linalg.solve()
        """
        self.X_fit_ = X
        gram = rbf_kernel(X, X, self.gamma) 
            # linalg solves Ax=b for x
        A = gram + self.alpha_reg * np.eye(X.shape[0])
        b = y
        self.dual_coef_ = np.linalg.solve(A, b)
        return self

    def predict(self, X):
        """
        1. Compute kernel k(X_fit, X).
        2. Return dot product with dual_coef_.
        """
        if self.X_fit_ is None:
            raise ValueError("Model not fitted.")
        
        kernel = rbf_kernel(self.X_fit_, X, gamma=self.gamma)  # shape (N_train, N_test)
        y_pred = kernel.T @ self.dual_coef_  # shape (N_test,)
        return y_pred
