import numpy as np

def polynomial_basis_matrix(X, degree):
    """
    Parameters
    ----------
    X : np.array [N,1]
        input array
    degree : int
        degree of the polynomial

    Returns
    -------
    Phi : np.array [N, degree + 1]
        polynomial basis matrix (includes bias column)
    """

    Phi = np.zeros((X.shape[0], degree + 1))
    for i in range(degree + 1):
        Phi[:, i] = np.squeeze(np.power(X, i))

    return Phi

class PolynomialBasis:
    """Callable basis object: basis(X) -> Phi."""
    def __init__(self, degree: int):
        self.degree = int(degree)

    def __call__(self, X):
        return polynomial_basis_matrix(X, self.degree)


class LeastSquares:
    """
    Ordinary Least Squares regression with arbitrary basis functions.
    
    Parameters
    ----------
    basis_function : callable
        Function that transforms input X into basis matrix Phi.
        Should accept X [N,1] and return Phi [N, M] where M is number of basis functions.
    """
    
    def __init__(self, basis_function):
        self.basis_function = basis_function
        self.weights = None
        self._is_fitted = False
    
    def fit(self, X, y):
        """
        Fit the least squares model.
        
        Parameters
        ----------
        X : np.array [N,1]
            Input features
        y : np.array [N,]
            Target values
            
        Returns
        -------
        self : LeastSquares
            Fitted model
        """
        # Convert to basis representation
        Phi = self.basis_function(X)
        
        # Ordinary least squares: w = (Phi^T Phi)^(-1) Phi^T y
        self.weights = np.linalg.solve(Phi.T @ Phi, Phi.T @ y)
        self._is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : np.array [N,1]
            Input features
            
        Returns
        -------
        y_pred : np.array [N,]
            Predicted values
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        Phi = self.basis_function(X)
        return Phi @ self.weights
    
    def score(self, X, y):
        """
        Compute mean squared error on test data.
        
        Parameters
        ----------
        X : np.array [N,1]
            Input features
        y : np.array [N,]
            True target values
            
        Returns
        -------
        mse : float
            Mean squared error
        """
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)


def mean_squared_error(y_true, y_pred):
    """
    Compute mean squared error between true and predicted values.
    
    Parameters
    ----------
    y_true : np.array [N,]
        True target values
    y_pred : np.array [N,]
        Predicted values
        
    Returns
    -------
    mse : float
        Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)


def generate_polynomial_data(n_samples=100, degree=3, noise_std=0.1, x_range=(-1, 1), random_state=42):
    """
    Generate synthetic polynomial data for testing.
    
    Parameters
    ----------
    n_samples : int
        Number of data points
    degree : int
        True polynomial degree
    noise_std : float
        Standard deviation of noise
    x_range : tuple
        Range of x values
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    X : np.array [n_samples, 1]
        Input features
    y : np.array [n_samples,]
        Target values
    """
    np.random.seed(random_state)
    
    # Generate random x values
    X = np.random.uniform(x_range[0], x_range[1], (n_samples, 1))
    
    # Generate polynomial coefficients
    coeffs = np.random.randn(degree + 1)
    
    # Compute polynomial values
    Phi = polynomial_basis_matrix(X, degree)
    y_true = Phi @ coeffs
    
    # Add noise
    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise
    
    return X, y

