import h5py
import numpy as np
from scipy import sparse
from pca_mdl2 import oja_pca
from pca import exact_pca, history_pca

from pathlib import Path
from time import time


def read_matrix(path: Path):
    with h5py.File(path, "r") as f:
        shape = f["shape"][()]
        data = f["data"][()]
        indices = f["indices"][()]
        indptr = f["indptr"][()]
    return sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=np.int32)


data_path = "/data1/intern/pca_benchmark/"
list_files = [
    "00.h5",
    "01.h5",
    "02.h5",
    "03.h5",
    "04.h5",
    "05.h5",
    "06.h5",
    "07.h5",
    "08.h5",
    "09.h5",
    "10.h5",
    "11.h5",
    "12.h5",
    "13.h5",
    "14.h5",
    "15.h5",
]

for file in list_files:
    f = open("multi_test.txt", "a")
    csr_mtx = read_matrix(data_path + file)
    print(csr_mtx.shape)
    csr_mtx = csr_mtx[:, :]
    print(f"{file} load success!")
    f.write(f"{file} load success!\n")

    k = 50

    t1 = time()
    oja_pca_mtx = oja_pca(csr_mtx, k)
    t2 = time()
    print("oja time: ", t2 - t1)
    f.write(f"oja time: {t2 - t1}\n")

    t1 = time()
    appro_pca_mtx = history_pca(csr_mtx, k=k)
    t2 = time()
    print("appro time: ", t2 - t1)
    f.write(f"appro time: {t2 - t1}\n")
    
    t1 = time()
    exact_pca_mtx = appro_pca_mtx #exact_pca(csr_mtx, k=k)
    t2 = time()
    print("exact time: ", t2 - t1)
    f.write(f"exact time: {t2 - t1}\n")

    exact_vars = np.var(exact_pca_mtx, axis=0)
    appro_vars = np.var(appro_pca_mtx, axis=0)
    oja_vars = np.var(oja_pca_mtx, axis=0)

    print("appro acc: ", np.sum(appro_vars) / np.sum(exact_vars))
    print("oja acc: ", np.sum(oja_vars) / np.sum(exact_vars))
    print("-" * 10)
    f.write(f"appro acc: {np.sum(appro_vars) / np.sum(exact_vars)}\n")
    f.write(f"oja acc: {np.sum(oja_vars) / np.sum(exact_vars)}\n")
    f.write("---------\n")

    f.close()