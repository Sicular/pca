from scipy.sparse.linalg import svds
from time import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from utils.utils import *
from scipy.sparse import csr_matrix


import sys
sys.path.insert(0, '/data1/intern/hoangnd/sparse_matrix/large_file_compress')
import read_large_h as csr


def pca_exact(X, k):
    mean = np.mean(X, axis=1, keepdims=True)
    std_input = X - mean
    S = std_input @ std_input.T / (X.shape[1]-1)
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
    print(w)
    return w.T.dot(X)


def oja_sbatch(X, k,batch_size = 2048, tol=1e-9):
    b = 1e-5
    c_time = 0
    m_time = 0
    a_time = 0
    gs_time = 0
    n_batch = int(np.ceil(X.shape[0]/batch_size))
    batch_size = min(batch_size,X.shape[0]//10)
    # his = []
    w = np.random.normal(0, 1, (X.shape[1], k))
    for j in range(n_batch):
        start = time()
        sX = csr_matrix(X[j * batch_size: (j + 1) * batch_size, :])
        c_time += time() -start
        # print("convert time: ",time()-start)
        
        start = time()
        G = 1 / (X.shape[0]) * sX.T @ (sX @ w)
        m_time += time()-start
        # print("multi time: ",time()-start)

        start =time()
        b = np.sqrt(b**2 + np.linalg.norm(G, axis=0) ** 2)
        a_time += time()-start
        # print("alpha time: ",time()-start)
        
        # print(f"alpha {j}: {1 / b}")
        start =time()
        w = gram_schmidt(w + (1 / b) * G)
        gs_time += time()-start
        # print("gs time: ",time()-start)

        # his.append(b)
    # f = plt.figure()
    # ax = f.add_subplot()
    # ax.plot(his)
    # f.savefig("./a.png")
    print(c_time,m_time,a_time,gs_time)
    return w.T @ X.T 


def oja_batch(X, k, max_iter, mu, dependency, tol=1e-9):
    b = 1e-5
    max_iter = 1
    batch_size = 1
    n_batch = int(np.ceil(X.shape[0]/batch_size))

    w = np.random.normal(0, 1, (X.shape[1], k))
    for i in range(max_iter):
        p = np.zeros((X.shape[1], X.shape[1]))
        for j in range(n_batch):
            G = (
                1
                / (X.shape[0])
                * X[j * batch_size: (j + 1) * batch_size, :].T
                @ X[j * batch_size: (j + 1) * batch_size, :]
            ) @ w
            b = np.sqrt(b**2 + np.linalg.norm(G, axis=0) ** 2)
            print(f"alpha: {1 / b}")
            w = gram_schmidt(w + (1 / b) * G)
    return w.T @ (X.T)


def oja_pp_method(X, k, max_iter, mu, dependency, tol=1e-9):
    C = X.dot(X.T) / X.shape[1]
    s = int(np.ceil(np.log2(k + 1)))
    w = None
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
        for j in range(max_iter):
            delta = alpha * C @ w
            old_w = w
            w = gram_schmidt(w + delta)
            if np.sum(np.abs(old_w - w)) < tol * w.size:
                print(f"{i} stop at {j}")
                break
    return w.T.dot(X)


def ada_oja_method(X, k, max_iter, mu, dependency, tol=1e-4):
    alpha = 1e-5
    w = np.random.normal(0, 1, (X.shape[0], k))
    w = gram_schmidt(w)
    C = X.dot(X.T) / X.shape[1]
    for i in range(max_iter):
        G = C @ w
        alpha = np.sqrt(alpha**2 + np.linalg.norm(G, axis=0) ** 2)
        print(f"alpha: {1 / alpha}")
        old_w = w
        w = gram_schmidt(w + (1 / alpha) * G)
        print(i)
    return w.T.dot(X)


dependency_kwargs = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 1]])
mu = 0, 0, 0
# scale = 6, 6, 1
k = 50
# input = get_correlated_dataset(500, dependency_kwargs, mu, scale, 3)
# input = get_random_dataset(10)
# input = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

input = csr.get_row(0, 109995)
input = input[:, :23636]
print(input.shape)

input = (input - input.mean(axis=0)) / np.std(input)

print("------------------")
start = time()
a, b, c = svds(input.astype("float"), k=k)
b = np.sort(b[:k] ** 2 / input.shape[0])[::-1]
print("svd check: ", b, "time: ", time()-start)

# print("------------------")
# start = time()
# eig_val, pca_ex = pca_exact(input.T, k)
# # check = np.abs(np.diag(np.cov(pca_ex)) / b[::-1]-1)
# print("exact: ", np.cov(pca_ex), "time: ")

# print("exact: ", np.diag(np.cov(pca_ex)), "time: ")
#       time()-start, "- Error: ", check)


# print("------------------")
# start = time()
# pca = oja_method(input.T, k, 1000, mu, dependency_kwargs)
# # check = np.abs(np.diag(np.cov(pca)) / b - 1)
# print("old: ", np.cov(pca), "time: ",)

# print("old: ", np.diag(np.cov(pca)), "time: ",
    #   time()-start, "- Error: ", check)

# print("------------------")
# start = time()
# pca = oja_pp_method(input, k, 1000, mu, dependency_kwargs)
# check = np.abs(np.sort(np.diag(np.cov(pca))) / b[::-1]-1)
# print("oja pp: ", np.sort(np.diag(np.cov(pca)))
#       [::-1], "time: ", time()-start, "- Error: ", check)


print("------------------")
start = time()
pca = oja_sbatch(input, k, 2048)
# check = np.abs(np.sort(np.diag(np.cov(pca)))[::-1] / b-1)
check = np.sum(np.diag(np.cov(pca))/np.sum(b))
print("batch: ", np.sort(np.diag(np.cov(pca)))[
      ::-1], "time: ", time()-start, "- Accuracy: ", check)


# print("exact eig: ", eig_val[:k])
