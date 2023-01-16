import numpy as np
from numpy import linalg as la
import h5py
from scipy import sparse
from pathlib import Path
from time import time
import sys

sys.path.append("/data1/intern/hanl/PCA/src/test")
from py.pca_benchmark import pca, exact_pca


def read_matrix(path: Path):
    with h5py.File(path, "r") as f:
        shape = f["shape"][()]
        data = f["data"][()]
        indices = f["indices"][()]
        indptr = f["indptr"][()]
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def DBPCA(matrix, k=50):
    s = np.ones((matrix.shape[1], k), dtype=matrix.dtype)
    q, _ = la.qr(s)
    block_size = min(8192, matrix.shape[0] // 10)
    n_blocks = (matrix.shape[1] + block_size - 1) // block_size
    for i in range(n_blocks):
        a = matrix[i * block_size : min((i + 1) * block_size, matrix.shape[1]), :]
        s += 1 / block_size * a.T @ (a @ q)
        q, _ = la.qr(s)
    return matrix @ q



