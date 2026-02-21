import numpy as np

class TwoHiddenLayerNN:
    def __init__(self, n_input, n_hidden1=32, n_hidden2=16, learning_rate=1e-3):
        """
        A very simple feed-forward neural network, with 2 hidden layers:
          input -> [Linear + ReLU] -> hidden1 -> [Linear + ReLU] -> hidden2 -> [Linear] -> output
        """
        # caches for backprop: you will need to store the values from the forward pass!
        self.cache = {}
        self._init_params(n_input, n_hidden1, n_hidden2)

    def _init_params(self, n_input, n_hidden1, n_hidden2):
        rng = np.random.RandomState(0)
        self.W1 = rng.randn(n_hidden1, n_input) * 0.01
        self.b1 = np.zeros((n_hidden1, 1))
        self.W2 = rng.randn(n_hidden2, n_hidden1) * 0.01
        self.b2 = np.zeros((n_hidden2, 1))
        self.W3 = rng.randn(1, n_hidden2) * 0.01
        self.b3 = np.zeros((1, 1))

    @staticmethod
    def relu(Z):
        return np.maximum(0,Z)
    
    @staticmethod
    def relu_grad(Z):
        return (Z>0).astype(float)

    def forward(self, X):
        """
        X: shape (n_input, m)
        Returns Y_hat: shape (1, m)
        """

        Z1 = self.W1 @ X + self.b1 
        H1 = self.relu(Z1)

        Z2 = self.W2 @ H1 + self.b2
        H2 = self.relu(Z2)

        Z3 = self.W3 @ H2 + self.b3 
        Y_hat = Z3

        self.cache = {"X": X, "Z1": Z1, "H1": H1,
                      "Z2": Z2, "H2": H2, "Z3": Z3, "Y_hat": Y_hat}
        
        return Y_hat

    def compute_loss(self, Y, Y_hat):
        """
        Y: (1, m), Y_hat: (1, m)
        returns scalar loss (mean squared error)
        """
        
        m = Y.shape[1] 

        loss = np.sum((Y_hat - Y)**2) / m

        
        return loss

    def backward(self, Y):
        """
        Y: (1, m)
        computes gradients w.r.t all parameters, returns dict grads
        """

        m = Y.shape[1]
        X = self.cache["X"]
        Z1, H1 = self.cache["Z1"], self.cache["H1"]
        Z2, H2 = self.cache["Z2"], self.cache["H2"]
        Y_hat = self.cache["Y_hat"]

        dZ3 = 2 * (Y_hat - Y) / m
        dW3 = dZ3.dot(H2.T)
        db3 = np.sum(dZ3, axis=1, keepdims=True)

       

        dH2 = self.W3.T @ dZ3
        dZ2 = dH2 * self.
        dW2 = dZ2.dot(H1.T)
        db2 = np.sum(dZ2, axis=1, keepdims=True)

        dH1 = self.W2.T @ dZ2
        dZ1 = dH1 * self.relu_grad(Z1)
        dW1 = dZ1.dot(X.T)
        db1 = np.sum(dZ1, axis=1, keepdims = True)

       

        return {"dW1": dW1, "db1": db1,
                "dW2": dW2, "db2": db2,
                "dW3": dW3, "db3": db3}

    def update_params(self, grads, optimizer):
        """
        Apply one optimizer step to all parameters
        """
        params = {"W1": self.W1, "b1": self.b1,
                  "W2": self.W2, "b2": self.b2,
                  "W3": self.W3, "b3": self.b3}
        updated = optimizer.step(params, grads)
        self.W1, self.b1 = updated["W1"], updated["b1"]
        self.W2, self.b2 = updated["W2"], updated["b2"]
        self.W3, self.b3 = updated["W3"], updated["b3"]