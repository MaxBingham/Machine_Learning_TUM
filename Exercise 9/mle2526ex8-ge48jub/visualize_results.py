import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from matplotlib.lines import Line2D

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Note: 'cartopy' not found. Borders will be approximated by data points.")

from sklearn.tree import DecisionTreeClassifier
from data_loader import load_housing_data

try:
    from knn_scratch import CustomKNNRegressor
except ImportError:
    print("Could not import CustomKNNRegressor. Make sure knn_scratch.py is implemented.")
    sys.exit(1)

def plot_california_boundary(model, X, y, title, is_classifier=False):
    """
    Trains a model on Lat/Long only and plots the decision surface.
    """
    print(f"Training {title} on location features (Lat/Long)...")
    
    X_loc = X[:, [0, 1]]
    
    model.fit(X_loc, y)
    
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES, linewidth=1, edgecolor='black')
        ax.add_feature(cfeature.COASTLINE, linewidth=1)
        ax.set_extent([-124.5, -114, 32, 42.5], crs=ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    x_min, x_max = -124.5, -114.0
    y_min, y_max = 32.0, 42.5
    
    h = 0.1 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    print(f"Predicting over {xx.size} mesh points...")
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    transform_args = {'transform': ccrs.PlateCarree()} if HAS_CARTOPY else {}
    
    if is_classifier:
        contour = ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], alpha=0.5, cmap=plt.cm.coolwarm, **transform_args)
        
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', label='Low Value (<= Median)',
                   markerfacecolor=plt.cm.coolwarm(0.0), markersize=10, alpha=0.5),
            Line2D([0], [0], marker='s', color='w', label='High Value (> Median)',
                   markerfacecolor=plt.cm.coolwarm(1.0), markersize=10, alpha=0.5)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
    else:
        contour = ax.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.viridis, **transform_args)
        
        cbar = plt.colorbar(contour, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label('Median House Value ($)')
        
    subset_idx = np.random.choice(len(X_loc), size=1000, replace=False)
    ax.scatter(X_loc[subset_idx, 0], X_loc[subset_idx, 1], 
               c='k', s=1, alpha=0.3, label='Training Data', **transform_args)
    
    ax.set_title(title)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize KNN or DT on California Data")
    parser.add_argument('mode', choices=['knn', 'dt'], help="Choose model to visualize")
    args = parser.parse_args()

    try:
        X_train, X_test, y_train, y_test, y_train_cls, y_test_cls = load_housing_data('housing.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if args.mode == 'knn':
        knn = CustomKNNRegressor(k=5, p=2, weight_mode='distance')
        plot_california_boundary(knn, X_train, y_train, 
                                 "k-NN Regression (Price Prediction)", 
                                 is_classifier=False)

    elif args.mode == 'dt':
        tree = DecisionTreeClassifier(max_depth=10, random_state=42)
        plot_california_boundary(tree, X_train, y_train_cls, 
                                 "Decision Tree (High vs Low Value)", 
                                 is_classifier=True)

if __name__ == "__main__":
    main()