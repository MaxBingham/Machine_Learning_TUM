import numpy as np

def entropy(y):
    """
    Compute Entropy of a label array.
    H(S) = - sum (p_i * log2(p_i))
    """
    if len(y) == 0: return 0.0
    # 1. Get counts of each class (np.unique return_counts=True)
    _, counts = np.unique(y, return_counts=True)
    # 2. Convert to probabilities
    probabilities = counts / len(y)
    # 3. Compute formula (add small epsilon to log if needed, or handle 0s)
    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def gini(y):
    """
    Compute Gini Impurity.
    G(S) = 1 - sum (p_i^2)
    """
    if len(y) == 0: return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

def information_gain(y_parent, y_left, y_right, metric='entropy'):
    """
    Compute Information Gain (IG).
    IG = Impurity(Parent) - [ (N_L/N_P)*Impurity(L) + (N_R/N_P)*Impurity(R) ]
    """
    # Select metric function based on 'metric' arg
    if metric == 'entropy':
        impurity_func = entropy
    else:
        impurity_func = gini
    
    # Calculate weighted average of child impurities
    n_parent = len(y_parent)
    n_left = len(y_left)
    n_right = len(y_right)
    
    parent_impurity = impurity_func(y_parent)
    left_impurity = impurity_func(y_left)
    right_impurity = impurity_func(y_right)
    
    weighted_child_impurity = (n_left / n_parent) * left_impurity + (n_right / n_parent) * right_impurity
    
    # Subtract from parent impurity
    return parent_impurity - weighted_child_impurity