import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def rectified_linear_unit(x):
    return np.maximum(0, x)
    