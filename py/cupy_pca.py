import numpy as np
import h5py
from scipy import sparse
from time import time
import cupy as cp
import cupyx as cpx

data_path = "/data1/intern/nhatnm/PCA/intern/pca_benchmark/"


def read_matrix(path):
    with h5py.File(path, 'r') as f:
        shape = f['shape'][()]
        data = np.array(f['data'][()])
        indices = np.array(f['indices'][()])
        indptr = np.array(f['indptr'][()])
    return sparse.csr_matrix((data, indices, indptr), shape=shape, dtype=np.int32)


def multi_cov_mat(a: sparse.csr_matrix, b, batch_sz=256):
    output = cp.zeros((a.shape[1], b.shape[1]))
    for start in range(0, a.shape[0], batch_sz):
        end = min(a.shape[0], start+batch_sz)
        mat_cp = cpx.scipy.sparse.csr_matrix(
            a[start:end],
            shape=(end-start, a.shape[1]),
            dtype=cp.float32
        )
        output += mat_cp.transpose()@(mat_cp @ b)
    return output


def multi_pca(a: sparse.csr_matrix, b, batch_sz=256):
    output = cp.zeros((a.shape[0], b.shape[1]))
    for start in range(0, a.shape[0], batch_sz):
        end = min(a.shape[0], start+batch_sz)
        mat_cp = cpx.scipy.sparse.csr_matrix(
            a[start:end],
            shape=(end-start, a.shape[1]),
            dtype=cp.float32
        )

        output[start:end] = mat_cp @ b
    return output


def power_method(csr_mat, k, batch):

    first_vectors_cp = cp.asarray(cp.random.rand(csr_mat.shape[1], k))
    first_vectors_cp, _ = cp.linalg.qr(first_vectors_cp)

    eig_mat = multi_cov_mat(csr_mat, first_vectors_cp, batch)

    eig_mat, _ = cp.linalg.qr(eig_mat)

    pca = multi_pca(csr_mat, eig_mat, batch)

    result = pca.get()

    return result


file = input("Data: ")
csr_mat = read_matrix(data_path + file)
print("Shape: ",csr_mat.shape)

k = int(input("Number of PC: "))
batch = int(input("Batch size: "))

print("Power methods: ")
t = time()
pca = power_method(csr_mat, k, batch)
print("\t", time() - t)

# np.save("rs.npy",pca)
