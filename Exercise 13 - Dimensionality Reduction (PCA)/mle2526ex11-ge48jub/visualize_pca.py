import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
import sys

# Ensure student code is importable
try:
    from principal_component_analysis import standardize, eigen_decomp_cov, project_data, get_variance_ratio, reconstruct_and_error
except ImportError:
    print("Could not import student code. Check PCA.py")
    sys.exit(1)

def main():
    # Load Data
    iris = load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    
    # 1. Standardize
    X_std, mean, std = standardize(X)
    
    # 2. Eigen Decomp
    eig_vals, eig_vecs = eigen_decomp_cov(X_std)
    
    # 3. Variance Ratio
    var_ratio = get_variance_ratio(eig_vals)
    
    # Setup Figures
    fig = plt.figure(figsize=(14, 6))
    
    # Left Plot: Interactive Scree Plot
    ax_scree = fig.add_subplot(1, 2, 1)
    
    # Right Plot: Scatter Plot (Dynamic)
    ax_scatter = fig.add_subplot(1, 2, 2) # Initial placeholder
    
    # State
    current_M = [2] # Mutable to allow access inside callback
    
    def update_plots(M):
        # Clear Right Plot
        fig.delaxes(fig.axes[1]) # Remove old scatter
        
        # Determine projection type
        if M == 1:
            ax_new = fig.add_subplot(1, 2, 2)
            projection_type = "1D"
        elif M == 2:
            ax_new = fig.add_subplot(1, 2, 2)
            projection_type = "2D"
        else: # M >= 3
            ax_new = fig.add_subplot(1, 2, 2, projection='3d')
            projection_type = "3D"
        
        # Project Data
        Z = project_data(X_std, eig_vecs, M=M)
        
        # Reconstruction Error
        mse = reconstruct_and_error(Z, eig_vecs, mean, std, X)
        
        # --- DRAW SCREE PLOT ---
        ax_scree.clear()
        x_ticks = range(1, len(var_ratio) + 1)
        ax_scree.plot(x_ticks, var_ratio, 'b-o', picker=5, label='Cumulative Variance')
        ax_scree.axvline(x=M, color='r', linestyle='--', alpha=0.5, label=f'Selected M={M}')
        ax_scree.set_title(f"Scree Plot (Click to Select M)\nCurrent MSE: {mse:.4f}")
        ax_scree.set_xlabel('Number of Components')
        ax_scree.set_ylabel('Explained Variance Ratio')
        ax_scree.set_xticks(x_ticks)
        ax_scree.set_ylim(0, 1.05)
        ax_scree.grid(True)
        ax_scree.legend()
        
        # --- DRAW SCATTER PLOT ---
        colors = ['navy', 'turquoise', 'darkorange']
        
        for color, i, target_name in zip(colors, [0, 1, 2], target_names):
            if M == 1:
                # 1D: Plot on x-axis, y is 0
                ax_new.scatter(Z[y == i, 0], np.zeros_like(Z[y == i, 0]), 
                               color=color, alpha=.8, label=target_name)
                ax_new.set_yticks([])
            elif M == 2:
                # 2D
                ax_new.scatter(Z[y == i, 0], Z[y == i, 1], 
                               color=color, alpha=.8, label=target_name)
            else:
                # 3D (Keep only top 3 dims for viz even if M=4)
                ax_new.scatter(Z[y == i, 0], Z[y == i, 1], Z[y == i, 2], 
                               color=color, alpha=.8, label=target_name)
                ax_new.set_zlabel('PC 3')
        
        ax_new.set_title(f"PCA Projection ({projection_type}) using top {M} components")
        ax_new.set_xlabel('Principal Component 1')
        if M > 1: ax_new.set_ylabel('Principal Component 2')
        ax_new.legend(loc='best')
        
        plt.draw()

    def on_pick(event):
        line = event.artist
        xdata = line.get_xdata()
        ind = event.ind
        # Get selected M (index + 1 because x-axis starts at 1)
        new_M = int(xdata[ind][0])
        current_M[0] = new_M
        print(f"Selected {new_M} components.")
        update_plots(new_M)

    # Connect event
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    # Initial Draw
    update_plots(2)
    
    print("Interactive Mode: Click on the blue dots in the Scree Plot to change components!")
    plt.show()

if __name__ == "__main__":
    main()