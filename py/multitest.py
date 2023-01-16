import h5py
import numpy as np
from scipy import sparse
from py.oja_method import oja_pca
from py.pca_benchmark import exact_pca, history_pca
from pca_pm import center_power_method
from DBPCA import DBPCA
from mypca import my_pca_new

import math
from scipy.stats import lognorm

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

for file in list_files[:1]:
    f = open("multi_test.txt", "a")
    csr_mtx = read_matrix(data_path + file)
    print(csr_mtx.shape)
    csr_mtx = csr_mtx[:, :]
    print(f"{file} load success!")
    f.write(f"{file} load success!\n")
    
    # sum_mtx = np.sum(csr_mtx, axis = 1) +1
    # new_mtx = csr_mtx / sum_mtx * 10000
    # csr_mtx = np.log1p(csr_mtx)
    np.random.seed(42)
    csr_mtx = lognorm.rvs(s=1, scale = math.exp(1), size=(csr_mtx.shape[0],csr_mtx.shape[1]))
    csr_mtx = sparse.csr_matrix(csr_mtx)
    print(csr_mtx.shape)

    k = 50

    t1 = time()
    oja_pca_mtx = oja_pca(csr_mtx, k, batch_sz=512)
    t2 = time()
    print("oja time: ", t2 - t1)
    f.write(f"oja time: {t2 - t1}\n")

    t1 = time()
    appro_pca_mtx = history_pca(csr_mtx, k=k)
    t2 = time()
    print("appro time: ", t2 - t1)
    f.write(f"appro time: {t2 - t1}\n")
    
    t1 = time()
    exact_pca_mtx = exact_pca(csr_mtx, k=k)
    t2 = time()
    print("exact time: ", t2 - t1)
    f.write(f"exact time: {t2 - t1}\n")

    exact_vars = np.var(exact_pca_mtx, axis=0)
    appro_vars = np.var(appro_pca_mtx, axis=0)
    oja_vars = np.var(np.array(oja_pca_mtx), axis=0)

    print("ext vars: ", np.sort(exact_vars)[::-1][:5] / np.max(exact_vars))
    print("oja vars: ", np.sort(oja_vars)[::-1][:5] / np.max(oja_vars))
    print("apr acc: ", np.sum(appro_vars) / np.sum(exact_vars))
    print("oja acc: ", np.sum(oja_vars) / np.sum(exact_vars))
    print("-" * 10)
    f.write(f"appro acc: {np.sum(appro_vars) / np.sum(exact_vars)}\n")
    f.write(f"oja acc: {np.sum(oja_vars) / np.sum(exact_vars)}\n")
    f.write("---------\n")

    f.close()