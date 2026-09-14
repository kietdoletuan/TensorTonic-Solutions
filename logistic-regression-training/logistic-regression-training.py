import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def _loss(y_hat, y_true):
    N = y_hat.shape[0]
    loss = -1/N * ((y_true*np.log(y_hat) + (1-y_true) * (np.log(1-y_hat))))

    return loss

def gradient_w(y_hat, y_true, X):
    return X.T @ (y_hat-y_true) / X.shape[0]

def gradient_b(y_hat, y_true):
    return np.mean(y_hat-y_true)

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    w = np.zeros(X.shape[1])
    b = 0

    for step in range(steps):
        z = X @ w + b
        y_hat = _sigmoid(z)

        print(f"{step}: {_loss(y_hat, y)}")

        grad_w = gradient_w(y_hat, y, X)
        grad_b = gradient_b(y_hat, y)

        w = w - lr*grad_w
        b = b - lr*grad_b

    return (w, b)
    
    pass