import numpy as np
from scipy import sparse


def normalize(x):
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n


def find_eigen_vetor(matrix):
    x = np.ones((matrix.shape[1], 1), dtype=matrix.dtype)
    for i in range(1000):
        x = np.dot(matrix, x)
        # x = matrix.T @ (matrix @ x)
        _, x = normalize(x)
    return x / np.linalg.norm(x)


def power_iteration(matrix, k=50):
    B = (sparse.csr_matrix.transpose(matrix)) @ matrix
    B = B.todense()
    l = list()
    for i in range(k):
        v = find_eigen_vetor(B)
        l.append(v)
        B = B - v @ np.dot(v.T, B)
    return matrix @ np.hstack(l)


def power_iteration_all_k(matrix, k):
    x = np.ones((matrix.shape[1], k), dtype=matrix.dtype)
    for i in range(100):
        x = matrix.T @ (matrix @ x)
        x, _ = np.linalg.qr(x)
    return matrix @ x 

