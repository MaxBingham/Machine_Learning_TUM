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

    X_stdt = np.transpose(X_std)
    Cov_Matrix = np.cov(X_stdt)

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
    # Select the top M eigenvectors
    top_eigenvectors = eigenvectors[:, :M]
    # Project the data
    Z = X_std @ top_eigenvectors
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
    # Normalize eigenvalues
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    # Cumulative sum
    cumulative_variance = np.cumsum(explained_variance_ratio)
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
    # Number of components used
    M = Z.shape[1]
    # Reconstruct in standardized space
    X_reconstructed_std = Z @ eigenvectors[:, :M].T
    # Unstandardize
    X_reconstructed = X_reconstructed_std * X_std_dev + X_mean
    # Compute MSE
    mse = np.mean((X_orig - X_reconstructed) ** 2)
    return mse


    # Main block to run demonstration code
    if __name__ == "__main__":
        # Example matrix for demonstration
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        
        # Task 1: Standardize
        X_std, mean, std = standardize(X)
        
        # Task 2: Eigen decomposition
        eigenvalues, eigenvectors = eigen_decomp_cov(X_std)
        print("Eigenvalues:", eigenvalues)
        print("Eigenvectors shape:", eigenvectors.shape)
        
        # Task 3: Project to top 2 components
        M = 2
        Z = project_data(X_std, eigenvectors, M)
        print("Projected data shape:", Z.shape)
        
        # Task 4: Variance ratio
        cumulative_variance = get_variance_ratio(eigenvalues)
        print("Cumulative variance ratio:", cumulative_variance)
        
        # Task 5: Reconstruct and compute error
        mse = reconstruct_and_error(Z, eigenvectors, mean, std, X)
        print("Reconstruction MSE:", mse)