"""Test script to verify all implemented functions work correctly."""
import numpy as np
from knn_scratch import minkowski_distance, compute_weights, CustomKNNRegressor
from tree_logic import entropy, gini, information_gain
from analysis_part1 import simulate_curse, tune_knn_hyperparameters
from tree_interp import train_and_analyze_pruning, get_top_features, get_tree_rules
from data_loader import load_housing_data

def test_distance_functions():
    print("Testing distance functions...")
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    
    # Test Euclidean distance (p=2)
    dist_euclidean = minkowski_distance(a, b, p=2)
    expected_euclidean = np.sqrt(27)  # sqrt((3^2 + 3^2 + 3^2))
    assert np.isclose(dist_euclidean, expected_euclidean), f"Expected {expected_euclidean}, got {dist_euclidean}"
    print(f"✓ Euclidean distance: {dist_euclidean:.4f}")
    
    # Test Manhattan distance (p=1)
    dist_manhattan = minkowski_distance(a, b, p=1)
    expected_manhattan = 9  # |3| + |3| + |3|
    assert np.isclose(dist_manhattan, expected_manhattan), f"Expected {expected_manhattan}, got {dist_manhattan}"
    print(f"✓ Manhattan distance: {dist_manhattan:.4f}")
    
    # Test compute_weights
    distances = np.array([1.0, 2.0, 3.0])
    weights = compute_weights(distances)
    assert np.isclose(np.sum(weights), 1.0), "Weights should sum to 1"
    assert np.all(weights > 0), "All weights should be positive"
    print(f"✓ Weights: {weights}")
    print()

def test_tree_functions():
    print("Testing tree functions...")
    
    # Test entropy
    y_perfect = np.array([0, 0, 0, 0])
    ent_perfect = entropy(y_perfect)
    assert np.isclose(ent_perfect, 0.0), f"Perfect split should have 0 entropy, got {ent_perfect}"
    print(f"✓ Entropy (perfect): {ent_perfect:.4f}")
    
    y_balanced = np.array([0, 0, 1, 1])
    ent_balanced = entropy(y_balanced)
    assert ent_balanced > 0, "Balanced split should have positive entropy"
    print(f"✓ Entropy (balanced): {ent_balanced:.4f}")
    
    # Test gini
    g_perfect = gini(y_perfect)
    assert np.isclose(g_perfect, 0.0), f"Perfect split should have 0 gini, got {g_perfect}"
    print(f"✓ Gini (perfect): {g_perfect:.4f}")
    
    g_balanced = gini(y_balanced)
    assert g_balanced > 0, "Balanced split should have positive gini"
    print(f"✓ Gini (balanced): {g_balanced:.4f}")
    
    # Test information gain
    y_parent = np.array([0, 0, 1, 1, 1, 1])
    y_left = np.array([0, 0])
    y_right = np.array([1, 1, 1, 1])
    ig = information_gain(y_parent, y_left, y_right, metric='entropy')
    assert ig >= 0, "Information gain should be non-negative"
    print(f"✓ Information gain: {ig:.4f}")
    print()

def test_knn_regressor():
    print("Testing CustomKNNRegressor...")
    
    # Create simple dataset
    X_train = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    y_train = np.array([1, 2, 3, 4])
    X_test = np.array([[2.5, 2.5]])
    
    # Test uniform weights
    knn = CustomKNNRegressor(k=2, p=2, weight_mode='uniform')
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    print(f"✓ KNN prediction (uniform): {pred[0]:.4f}")
    
    # Test distance weights
    knn_dist = CustomKNNRegressor(k=2, p=2, weight_mode='distance')
    knn_dist.fit(X_train, y_train)
    pred_dist = knn_dist.predict(X_test)
    print(f"✓ KNN prediction (distance): {pred_dist[0]:.4f}")
    print()

def test_curse_of_dimensionality():
    print("Testing curse of dimensionality simulation...")
    ratios = simulate_curse(d_values=[2, 10, 50], n_samples=100)
    assert len(ratios) == 3, "Should have 3 dimension results"
    for d, ratio in ratios.items():
        print(f"✓ Dimension {d}: ratio = {ratio:.4f}")
    print()

def test_with_real_data():
    print("Testing with real housing data...")
    try:
        X_train, X_test, y_train, y_test, y_train_cls, y_test_cls = load_housing_data('housing.csv')
        print(f"✓ Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        # Test KNN on small subset
        subset_size = 500
        X_subset = X_train[:subset_size]
        y_subset = y_train[:subset_size]
        
        knn = CustomKNNRegressor(k=5, p=2, weight_mode='distance')
        knn.fit(X_subset, y_subset)
        predictions = knn.predict(X_test[:10])
        print(f"✓ KNN made predictions: {predictions[:3]}")
        
        # Test tree functions
        from sklearn.tree import DecisionTreeClassifier
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree.fit(X_subset, y_train_cls[:subset_size])
        
        feature_names = ['Longitude', 'Latitude', 'HousingMedianAge', 'TotalRooms',
                        'TotalBedrooms', 'Population', 'Households', 'MedianIncome']
        top_features = get_top_features(tree, feature_names, top_n=3)
        print(f"✓ Top features: {[name for name, _ in top_features]}")
        
        print()
    except FileNotFoundError:
        print("⚠ housing.csv not found, skipping real data tests")
        print()

if __name__ == "__main__":
    print("="*60)
    print("Running implementation tests...")
    print("="*60)
    print()
    
    test_distance_functions()
    test_tree_functions()
    test_knn_regressor()
    test_curse_of_dimensionality()
    test_with_real_data()
    
    print("="*60)
    print("✅ All tests completed successfully!")
    print("="*60)
