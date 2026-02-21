import numpy as np

def select_next_batch(model, X_pool, k=1):
    """
    Task 6: Acquisition Function (Max Variance).
    
    Args:
        model: An object with a predict_uncertainty(X) method.
        X_pool: The pool of unlabeled data (N, D).
        k: Number of samples to select.
        
    Returns:
        indices: The indices of the k samples in X_pool with highest uncertainty.
    """
    # 1. Get uncertainties (std for each sample)
    _, sigma = model.predict_uncertainty(X_pool)
    uncertainties = []
    for i in range(len(sigma)):
        uncertainties.append((sigma[i], i))

    # 2. Sort by uncertainty descending
    uncertainties.sort(key=lambda t: t[0], reverse=True)

    # 3. Return top k indices
    top_k_indices = [idx for (_, idx) in uncertainties[:k]]
    return top_k_indices

def active_learning_loop(model, X_pool, y_pool_oracle, X_init, y_init, n_steps=10, batch_size=1):
    """
    Task 7: The Active Learning Loop.
    
    Yields:
        (model, X_train, indices_selected) at each step.
    """
    # Initialize training sets
    X_train = X_init.copy()
    y_train = y_init.copy()
    
    # Mask to keep track of available pool points
    # (True = available, False = already labeled)
    pool_mask = np.ones(len(X_pool), dtype=bool)

    for step in range(n_steps):
        if np.sum(pool_mask) == 0: break
        
        # 1. Fit Model on current X_train
        model.fit(X_train, y_train)
        
        
        # 2. Select next batch from AVAILABLE pool
        # Hint: Pass only X_pool[pool_mask] to selection function
        available_indices = np.where(pool_mask)[0]
        X_candidates = X_pool[available_indices]
        
        # Get relative indices from select_next_batch
        relative_indices = select_next_batch(model, X_candidates, k=batch_size)
        
        # Convert to global indices in X_pool
        selected_global_indices = available_indices[relative_indices]
        
        # Simulate oracle query (get y from y_pool_oracle)
        y_oracle = y_pool_oracle[selected_global_indices]
        
        # Update X_train, y_train
        X_train = np.vstack([X_train, X_pool[selected_global_indices]])
        y_train = np.concatenate([y_train, y_oracle])
        
        # Update pool_mask
        pool_mask[selected_global_indices] = False
        
        yield model, X_train, selected_global_indices