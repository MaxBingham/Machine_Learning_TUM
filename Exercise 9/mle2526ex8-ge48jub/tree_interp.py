import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

def train_and_analyze_pruning(X_train, y_train, X_val, y_val, max_depths):
    """
    Train trees with different max_depths.
    Return:
        train_accs: list of training accuracies
        val_accs: list of validation accuracies
    """
    train_accs = []
    val_accs = []
    
    # Loop over max_depths
    for depth in max_depths:
        # 1. Fit DecisionTreeClassifier (random_state=42)
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        
        # 2. Predict on train and val
        y_train_pred = tree.predict(X_train)
        y_val_pred = tree.predict(X_val)
        
        # 3. Store accuracy_score
        train_accs.append(accuracy_score(y_train, y_train_pred))
        val_accs.append(accuracy_score(y_val, y_val_pred))
    
    return train_accs, val_accs

def get_top_features(model, feature_names, top_n=3):
    """
    Extract feature importances from the model and return the top_n features.
    Returns: list of tuples [(feature_name, importance_score), ...] sorted descending.
    """
    # Access model.feature_importances_
    importances = model.feature_importances_
    
    # Pair with feature_names
    feature_importance_pairs = list(zip(feature_names, importances))
    
    # Sort and slice
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    return feature_importance_pairs[:top_n]

def get_tree_rules(model, feature_names):
    """
    Return the text representation of the decision tree rules.
    """
    # Use sklearn.tree.export_text
    return export_text(model, feature_names=feature_names)