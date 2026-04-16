import numpy as np
from tqdm import tqdm

class SoftSVM:
    def __init__(self, lr = 0.001, C=1.0, epochs=1000):
        self.w = None
        self.b = None
        self.lr = lr
        self.C = C
        self.epochs = epochs
        self.losses = []

    def fit(self, X, y):
        n_samples, n_features = X.shape # X (N, d)
        self.w = np.zeros(n_features)
        self.b = 0.0

        pbar = tqdm(range(self.epochs), desc="Training")

        for epoch in pbar:
            idx = np.random.permutation(n_samples)
            X, y = X[idx], y[idx]

            for i in range(n_samples):
                xi = X[i]
                yi = y[i]

                if yi * (xi @ self.w + self.b) < 1:
                    dw = self.w - self.C * yi * xi
                    db = -self.C * yi

                else:
                    dw = self.w
                    db = 0.0

                self.w -= self.lr * dw
                self.b -= self.lr * db

            loss = self.hinge_loss(X, y)
            self.losses.append(loss)
            pbar.set_postfix(loss=f"{loss:.4f}")
            
    def predict(self, X):
        return np.sign(X @ self.w + self.b)
    
    def hinge_loss(self, X, y):
        return 1 / 2 * np.dot(self.w, self.w) + self.C * np.sum(np.maximum(0, 1 - y * (X @ self.w + self.b)))
