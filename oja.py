import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from utils.utils import *
from scipy.sparse import csr_matrix


def pca_exact(X, k):
    mean = np.mean(X, axis=1, keepdims=True)
    std_input = X - mean
    S = std_input @ std_input.T / (X.shape[1])
    _, eigen_vector = np.linalg.eig(S)
    Uk = eigen_vector[:, :k]
    return _, np.dot(Uk.T, std_input)


def oja_method(X, k, max_iter, mu, dependency, tol=1e-9):
    alpha = 0.5
    w = np.random.normal(0, 1, (X.shape[0], k))
    w = gram_schmidt(w)
    C = X.dot(X.T) / X.shape[1]
    for i in range(max_iter):
        w = gram_schmidt(w + alpha * C @ w)
    # display3D(X.T, mu, w.T)
    # display2D(X.T, mu, w.T)
    return w.T.dot(X)


def oja_batch(X, k, max_iter=1, batch_size=1, tol=1e-9):
    b = 1e-5
    n_batch = int(np.ceil(X.shape[0] / batch_size))
    w = np.random.normal(0, 1, (X.shape[1], k))
    his = []
    for i in range(max_iter):
        p = np.zeros((X.shape[1], X.shape[1]))
        for j in range(n_batch):
            x = csr_matrix(X[j * batch_size : (j + 1) * batch_size, :])
            G = (1 / (X.shape[0]) * x.T @ x) @ w
            # b = np.sqrt(b**2 + np.linalg.norm(G, axis=0) ** 2)
            w = gram_schmidt(w + (1 / b) * G)
            his.append(b)

    f = plt.figure()
    ax = f.add_subplot()
    ax.plot(his)
    f.savefig("./a.png")
    return X @ w


def oja_pp_method(X, k, max_iter, mu, dependency, tol=1e-5):
    C = X.dot(X.T) / X.shape[1]
    s = int(np.ceil(np.log2(k + 1)))
    w = None
    T0 = 500
    for i in range(s):
        if i == 0:
            w = np.random.normal(0, 1, (X.shape[0], k - k // 2))
        else:
            w = np.hstack(
                (
                    w,
                    np.random.normal(
                        0, 1, (X.shape[0], k // (2**i) - k // 2 ** (i + 1))
                    ),
                )
            )
        w = gram_schmidt(w)
        alpha = 1
        for j in range(T0):
            # G = C @ w
            # alpha = np.sqrt(alpha**2 + np.linalg.norm(G, axis=0) ** 2)
            delta = alpha * C @ w
            # print(alpha)
            old_w = w
            w = gram_schmidt(w + delta)
            if np.sum(np.abs(old_w - w)) < tol * w.size:
                print(f"{i} stop at {j}")
                break
            display3D(X.T, mu, w.T, animation=True)

    plt.show()
    return w.T.dot(X)


dependency_kwargs = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 1]])
mu = 0, 0, 0
scale = 6, 6, 1
k = 10
input = get_correlated_dataset(500, dependency_kwargs, mu, scale, 3)
# input = get_random_dataset(1000)
import utils.csr_matrix as csr
input = csr.get_multirow(0, 1000)[:,:]
# display3D(input, mu, dependency_kwargs)
print(input)
# input = np.array([x, y, z])
# input = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
input = (input - input.mean(axis=0)) / np.std(input)

print("mean ", input.mean())
print("std ", np.std(input))
print(input.shape)
# pca = oja_method(input.T, k, 1000, mu, dependency_kwargs)
# print("old: ", np.diag(np.cov(pca)))

from time import time
import _oja
t1 = time()
pca = input @ _oja.oja_batch(input.shape[0], input.shape[1], k, 1, 1, float(0.01))
t2 = time()
print(t2 - t1)
# print(pca.shape)
# print("new: ", np.cov(pca.T))
print("new: ", np.sort(np.diag(np.cov(pca.T))))

from scipy.sparse.linalg import svds
t1 = time()
a, b, c = svds(input.astype("float"), k=k)
b = b[:k] ** 2 / (input.shape[0])
t2 = time()
print(t2 - t1)
print("svd: ", b)