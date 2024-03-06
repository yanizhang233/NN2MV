import numpy as np
def sigma(x):
    return np.minimum(np.maximum(x, 0), 1)

def relu(x):
    return np.maximum(x, 0)