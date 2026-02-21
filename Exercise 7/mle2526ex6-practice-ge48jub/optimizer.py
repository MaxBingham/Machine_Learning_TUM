import numpy as np

class RMSPropOptimizer:
    def __init__(self, lr=1e-3, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = {}

    def step(self, params, grads):
        updated = {}
        for name, theta in params.items():
            g = grads["d"+name]

            self.s[name] = self.beta * self.s.get(name, 0) + (1 - self.beta) * (g ** 2)
            theta_new = theta - self.lr * g / (np.sqrt(self.s[name]) + self.eps)

            updated[name] = theta_new
        return updated

class AdamOptimizer:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, params, grads):
        self.t += 1
        updated = {}
        for name, theta in params.items():
            g = grads["d"+name]

            self.m[name] = self.b1 * self.m.get(name, 0) + (1 - self.b1) * g
            self.v[name] = self.b2 * self.v.get(name, 0) + (1 - self.b2) * (g ** 2)
            m_hat = self.m[name] / (1 - self.b1 ** self.t)
            v_hat = self.v[name] / (1 - self.b2 ** self.t)
            theta_new = theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            updated[name] = theta_new
        return updated
