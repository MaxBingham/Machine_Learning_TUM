import argparse
import numpy as np
import matplotlib.pyplot as plt
from neural_network_scratch import TwoHiddenLayerNN
from optimizer import RMSPropOptimizer, AdamOptimizer

LEARNING_RATE = 0.01
EPOCHS = 1000
BATCH_SIZE = 64
VERBOSITY = 50

# This function should be fixed, it's this version that will be used in the tests.
def load_data():
    np.random.seed(0)
    X = np.random.uniform(-20, 20, (1, 2000))
    y = np.sin(X) / X
    y[X == 0] = 1.0
    y = y.reshape(1, -1)

    # split into training and validation sets
    split_idx = int(0.8 * X.shape[1])
    X_train, y_train = X[:, :split_idx], y[:, :split_idx]
    X_val, y_val = X[:, split_idx:], y[:, split_idx:]
    return X_train, y_train, X_val, y_val

def plot_predictions(X, model, optimizer):
    X_full = np.linspace(-20, 20, 1000).reshape(1, -1)
    y_full = np.sin(X_full) / X_full
    y_full[X_full == 0] = 1.0
    y_hat_val = model.forward(X)
    y_hat_val = y_hat_val
    plt.figure(figsize=(10, 5))
    plt.plot(X_full.flatten(), y_full.flatten(), label='True Function', color='blue', linewidth=0.5)
    plt.scatter(X.flatten(), y_hat_val.flatten(), label='Predictions', color='red', s=10)
    plt.title(f"Final Predictions vs True Function ({optimizer.__class__.__name__})")
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def train_scratch(X_train, y_train, X_val, y_val, optimizer, plot=True):
    model = TwoHiddenLayerNN(n_input=X_train.shape[0])
    optimizer = optimizer(lr=LEARNING_RATE)
    for epoch in range(1, EPOCHS+1):
        perm = np.random.permutation(X_train.shape[1])
        X_sh, y_sh = X_train[:, perm], y_train[:, perm]
        for i in range(0, X_train.shape[1], BATCH_SIZE):
            X_batch = X_sh[:, i:i+BATCH_SIZE]
            y_batch = y_sh[:, i:i+BATCH_SIZE]
            
            ###TODO####
            ###########
            
            ###########
            ############
        if epoch % VERBOSITY == 0:
            y_hat_train = model.forward(X_train)
            y_hat_val = model.forward(X_val)
            loss_train = model.compute_loss(y_train, y_hat_train)
            loss_val = model.compute_loss(y_val, y_hat_val)
            print(f"[Scratch][{optimizer.__class__.__name__}] Epoch {epoch}: train_loss={loss_train:.4f}, val_loss={loss_val:.4f}")
    
    if plot:
        plot_predictions(X_val, model, optimizer)
    
    return model 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["rmsprop","adam","both"], default="both")
    parser.add_argument("--plot", action='store_true', help="Plot predictions after training")
    args = parser.parse_args()

    X_train, y_train, X_val, y_val = load_data()

    for opt in ([args.optimizer] if args.optimizer!="both" else ["rmsprop","adam"]):
        train_scratch(X_train, y_train, X_val, y_val,
                RMSPropOptimizer if opt=='rmsprop' else AdamOptimizer, args.plot)


if __name__ == "__main__":
    main()