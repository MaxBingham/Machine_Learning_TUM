import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

try:
    from analysis_part1 import simulate_curse
    from tree_interp import train_and_analyze_pruning, get_tree_rules, get_top_features
    from data_loader import load_housing_data
except ImportError:
    print("Could not import required modules. Make sure all files are implemented.")
    sys.exit(1)

def plot_curse_of_dimensionality():
    print("Simulating Curse of Dimensionality...")
    d_values = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ratios = simulate_curse(d_values, n_samples=1000)
    
    dims = sorted(ratios.keys())
    vals = [ratios[d] for d in dims]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dims, vals, marker='o', linestyle='-', color='r')
    plt.yscale('log')
    plt.xlabel('Dimensions (d)')
    plt.ylabel('Contrast Ratio (max-min)/min')
    plt.title('The Curse of Dimensionality: Distance Contrast vs Dimension')
    plt.grid(True, which="both", ls="-")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.show()

def plot_tree_overfitting():
    print("Loading data for Overfitting Analysis...")
    X_train, X_test, _, _, y_train_cls, y_test_cls = load_housing_data('housing.csv')
    
    depths = range(1, 21)
    print(f"Training trees with depths {list(depths)}...")
    train_accs, val_accs = train_and_analyze_pruning(X_train, y_train_cls, X_test, y_test_cls, depths)
    
    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_accs, label='Training Accuracy', marker='o')
    plt.plot(depths, val_accs, label='Validation Accuracy', marker='s')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Decision Tree Overfitting: Depth vs Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(depths)
    plt.show()

def show_tree_interpretation():
    print("Training interpretability tree...")
    X_train, _, _, _, y_train_cls, _ = load_housing_data('housing.csv')
    
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train_cls)
    
    feature_names = [
        'Longitude', 'Latitude', 'HousingMedianAge', 'TotalRooms',
        'TotalBedrooms', 'Population', 'Households', 'MedianIncome'
    ]
    
    print("\n--- TOP FEATURES ---")
    top_feats = get_top_features(model, feature_names, top_n=5)
    for name, score in top_feats:
        print(f"{name}: {score:.4f}")
        
    print("\n--- TREE RULES ---")
    print(get_tree_rules(model, feature_names))

def main():
    parser = argparse.ArgumentParser(description="Visualize Analysis (Curse, Overfitting, Rules)")
    parser.add_argument('mode', choices=['curse', 'overfitting', 'rules'], help="Choose analysis to run")
    args = parser.parse_args()

    if args.mode == 'curse':
        plot_curse_of_dimensionality()
    elif args.mode == 'overfitting':
        plot_tree_overfitting()
    elif args.mode == 'rules':
        show_tree_interpretation()

if __name__ == "__main__":
    main()