import numpy as np
import activations

def derivative_sigmoid(x):
    s = activations.sigmoid(x)
    return s * (1 - s)

def derivative_rectified_linear_unit(x):
    return np.where(x > 0, 1, 0)

def binary_cross_entropy(y_true, y_pred):
    """
    y_true: Actual labels (numpy array of shape (n,))
    y_pred: Predicted probabilities (numpy array of shape (n,))
    Returns: Binary cross-entropy loss
    """
    # Clip predicted values to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss