import numpy as np

def standardize(X):
    """
    Task 1: Standardize the dataset.
    Z = (X - mean) / std
    
    Args:
        X: Dataset of shape (N, D)
    Returns:
        X_std: Standardized data (N, D)
        mean: Mean vector (D,)
        std: Standard deviation vector (D,)
    """
    mean = None
    std = None
    X_std = None
    
    #Mistake - didnt add axis --> Outputs scalar value  
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0) 
    std = np.where(std == 0, 1, std)  # Avoid division by zero for constant columns

    Z = (X - mean) / std 
    X_std = Z
    
    print("Standardized data (X_std):")
    print(X_std)
    
    return X_std, mean, std

def eigen_decomp_cov(X_std):
    """
    Task 2: Eigen Decomposition of Covariance Matrix.
    1. Compute covariance matrix of X_std.
    2. Compute eigenvalues and eigenvectors.
    3. Sort them in descending order of eigenvalues.
    
    Args:
        X_std: Standardized data (N, D)
    Returns:
        eigenvalues: Sorted eigenvalues (D,)
        eigenvectors: Sorted eigenvectors (D, D) as columns
    """
    eigenvalues = None
    eigenvectors = None


    Cov_Matrix = np.cov(X_std.T)

    eigenvalues, eigenvectors = np.linalg.eig(Cov_Matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

def project_data(X_std, eigenvectors, M):
    """
    Task 3: Dimensionality Reduction.
    Project data onto the top M eigenvectors.
    
    Args:
        X_std: Standardized data (N, D)
        eigenvectors: Sorted eigenvectors (D, D)
        M: Number of principal components to keep
    Returns:
        Z: Projected data (N, M)
    """
    
    EV = eigenvectors[:, :M]

    Z = X_std @ EV
    
    return Z

def get_variance_ratio(eigenvalues):
    """
    Task 4: Explained Variance.
    Compute the cumulative explained variance ratio.
    
    Args:
        eigenvalues: Sorted eigenvalues (D,)
    Returns:
        cumulative_variance: Array of shape (D,) where the i-th element 
                             is the sum of the first i+1 normalized eigenvalues.
    """

    explained_variance = eigenvalues/np.sum(eigenvalues)

    cumulative_variance = np.cumsum(explained_variance)
    
    return cumulative_variance

def reconstruct_and_error(Z, eigenvectors, X_mean, X_std_dev, X_orig):
    """
    Task 5: Reconstruction and Error.
    1. Reconstruct data from Z back to original space.
    2. Compute MSE between original X and reconstructed X.
    
    Args:
        Z: Projected data (N, M)
        eigenvectors: Sorted eigenvectors (D, D)
        X_mean: (D,) from standardization
        X_std_dev: (D,) from standardization
        X_orig: Original data (N, D)
    Returns:
        mse: Mean Squared Error (scalar)
    """

    M = Z.shape[1]

    X_reconstructed = Z @ eigenvectors[:,:M].T
    X_reconstructed = (X_reconstructed * X_std_dev) + X_mean

    mse = np.square(np.subtract(X_orig, X_reconstructed)).mean()

    return mse 
