import numpy as np
def removeRedundant(W, B, W_next):
    n = W.shape[0]
    hashMap = {}
    for i in range(n):
        new = np.concatenate((W[i], B[i]))
        new = str(new)
        if new not in hashMap:
            hashMap[new] = i
        else:
            W_next[:, hashMap[new]] += W_next[:, i]
            W_next[:, i] = 0
    keepCol = W_next.any(0)

    return [W[keepCol,:], B[keepCol,:], W_next[:, keepCol]]
