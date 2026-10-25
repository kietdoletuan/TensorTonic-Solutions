import numpy as np

def linear_regression_from_scratch(X: list, y: list, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    w = np.zeros(X.shape[1])
    b = 0

    print(w.shape)
    
    for epoch in range(epochs):
        y_hat = X @ w + b
        

        error = y_hat - y

        grad_w = (2/n) * (X.T @ error)
        grad_b = (2/n) * np.sum(error, axis=0)

        w -= lr * grad_w
        b -= lr * grad_b

    return [w.tolist(), b.tolist()]
    
    pass
