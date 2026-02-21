import matplotlib.pyplot as plt
import numpy as np
import argparse
from data_loader import load_housing_data
from ensembles import train_rf_regressor, train_adaboost_classifier, get_staged_scores

def plot_map(mode):
    (X_train, X_test, y_train, y_test, y_train_cls, y_test_cls), _ = load_housing_data()
    X_loc = X_train[:, [0, 1]] # Longitude, Latitude
    
    if mode == 'knn': # For visual comparison with Ex 8
        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=5).fit(X_loc, y_train)
        title = "Baseline: k-NN Regression"
    elif mode == 'rf':
        model = train_rf_regressor(X_loc, y_train, n_estimators=10)
        title = "Bagging: Random Forest Regression"
    else:
        model = train_adaboost_classifier(X_loc, y_train_cls, n_estimators=50)
        title = "Boosting: AdaBoost Classification"

    h = 0.1
    xx, yy = np.meshgrid(np.arange(-124.5, -114, h), np.arange(32, 42.5, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.7, cmap='coolwarm' if 'Class' in title else 'viridis')
    plt.title(title)
    plt.show()

def plot_curve():
    (X_train, X_test, y_train, y_test, y_train_cls, y_test_cls), _ = load_housing_data()
    model = train_adaboost_classifier(X_train, y_train_cls, n_estimators=100)
    scores = get_staged_scores(model, X_test, y_test_cls)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 101), scores, label='AdaBoost Test Accuracy')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Accuracy')
    plt.title('AdaBoost Staged Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['rf', 'ada', 'curve'], default='rf')
    args = parser.parse_args()
    if args.mode == 'curve': plot_curve()
    else: plot_map(args.mode)