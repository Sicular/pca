import h5py
import numpy as np
from scipy import sparse
from oja_method import oja_batch, oja_batch_cupy, oja_pca
from pca_benchmark import exact_pca, history_pca
import cupy as cp

# from _oja import oja_batch
from pathlib import Path
from time import time


def read_matrix(path: Path):
    with h5py.File(path, "r") as f:
        shape = f["shape"][()]
        data = f["data"][()]
        indices = f["indices"][()]
        indptr = f["indptr"][()]
    return sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32)


data_path = "/data1/iamb/pca/data/"
csr_mtx = read_matrix(data_path + "GSE154826.hdf5")
print("load success!")
print(f"data shape: {csr_mtx.shape}\n{'-' * 20}")
# Log norm
# sum_mtx = np.sum(csr_mtx, axis = 1) + 1
# new_mtx = csr_mtx / sum_mtx * 10000
# csr_mtx = sparse.csr_matrix(np.log1p(new_mtx))

k = 50

t1 = time()
oja_pca_mtx = oja_pca(csr_mtx, k)
t2 = time()
print("oja time: ", t2 - t1)

t1 = time()
appro_pca_mtx = history_pca(csr_mtx, k=k)
t2 = time()
print("appro time: ", t2 - t1)

t1 = time()
exact_pca_mtx = appro_pca_mtx #exact_pca(csr_mtx, k=k)
t2 = time()
print("exact time: ", t2 - t1)
print("-" * 20)

exact_vars = np.var(exact_pca_mtx, axis=0)
print("ext vars: ", np.sort(exact_vars)[::-1][:5] / np.max(exact_vars))
appro_vars = np.var(appro_pca_mtx, axis=0)
oja_vars = np.var(oja_pca_mtx, axis=0)

np.savetxt("appro_vars.txt", appro_vars)
np.savetxt("oja_vars.txt", oja_vars)
print("oja vars: ", np.sort(oja_vars)[::-1][:5] / np.max(oja_vars))
print("appro acc: ", np.sum(appro_vars) / np.sum(exact_vars))
print("oja acc: ", np.sum(oja_vars) / np.sum(exact_vars))
print("-" * 20)
print("exact vars: ", np.sum(exact_vars))
print("appro vars: ", np.sum(appro_vars))
print("oja vars: ", np.sum(oja_vars))
